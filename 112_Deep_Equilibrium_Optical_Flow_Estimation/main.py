from __future__ import division, print_function

import sys

sys.path.append('core')

import argparse
import os
import time
from functools import partial

import cv2
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import evaluate, viz
from metrics import compute_epe,  merge_metrics

from deq import get_model
# from deq_demo import get_model
# For a simple demo, please refer to deq_demo.py

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
TIME_FREQ = 500


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, cal_epe=True):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
    
    if cal_epe:
        epe = compute_epe(flow_preds[-1], flow_gt, valid)
        return flow_loss, epe
    else:
        return flow_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    if args.schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_steps, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = args.resume_iter if args.resume_iter > 0 else 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        sorted_keys = sorted(self.running_loss.keys())
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted_keys]
        training_str = "[Step {:6d}, lr {:.7f}]   ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ", ".join([f"{name}:{val:10.4f}" for (name, val) in zip(sorted_keys, metrics_data)])
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter("runs/" + args.name)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    DEQFlow = get_model(args)
    model = nn.DataParallel(DEQFlow(args), device_ids=args.gpus)
    print("Parameter Count: %.3f M" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    scheduler.last_epoch = args.resume_iter if args.resume_iter > 0 else -1

    total_steps = args.resume_iter if args.resume_iter > 0 else 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    add_noise = True
    best_chairs = {"epe": 1e8}
    best_sintel = {"clean-epe": 1e8, "final-epe": 1e8}
    best_kitti = {"epe": 1e8, "f1": 1e8}
    should_keep_training = True
    while should_keep_training:
        
        timer = 0

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            
            start_time = time.time()

            step_seq_loss = partial(sequence_loss, gamma=args.gamma)
            loss, metrics = model(
                image1, image2, flow, valid, step_seq_loss,
                )            

            metrics = merge_metrics(metrics)
            scaler.scale(loss.mean()).backward()
            
            end_time = time.time()
            timer += end_time - start_time

            scaler.unscale_(optimizer)   
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            
            if (total_steps + 1) % args.time_interval == 0:
                print(f'Exp {args.name} Average Time: {timer / args.time_interval}')
                timer = 0
            
            if (total_steps + 1) % args.save_interval == 0:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

            if total_steps % args.eval_interval == args.eval_interval - 1:
                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        res = evaluate.validate_chairs(model.module, sradius_mode=args.sradius_mode, best=best_chairs)
                        best_chairs['epe'] = min(res['chairs'], best_chairs['epe'])
                        results.update(res)
                    elif val_dataset == 'things':
                        results.update(evaluate.validate_things(model.module, sradius_mode=args.sradius_mode))
                    elif val_dataset == 'sintel':
                        res = evaluate.validate_sintel(model.module, sradius_mode=args.sradius_mode, best=best_sintel)
                        best_sintel['clean-epe'] = min(res['clean'], best_sintel['clean-epe'])
                        best_sintel['final-epe'] = min(res['final'], best_sintel['final-epe'])
                        results.update(res)
                    elif val_dataset == 'kitti':
                        res = evaluate.validate_kitti(model.module, sradius_mode=args.sradius_mode, best=best_kitti)
                        best_kitti['epe'] = min(res['kitti-epe'], best_kitti['epe'])
                        best_kitti['f1'] = min(res['kitti-f1'], best_kitti['f1'])
                        results.update(res)

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


def val(args):
    DEQFlow = get_model(args)
    model = nn.DataParallel(DEQFlow(args), device_ids=args.gpus)
    print("Parameter Count: %.3f M" % count_parameters(model))
    
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.eval()
    
    for val_dataset in args.validation:
        if val_dataset == 'chairs':
            evaluate.validate_chairs(model.module, sradius_mode=args.sradius_mode)
        elif val_dataset == 'things':
            evaluate.validate_things(model.module, sradius_mode=args.sradius_mode)
        elif val_dataset == 'sintel':
            evaluate.validate_sintel(model.module, sradius_mode=args.sradius_mode)
        elif val_dataset == 'kitti':
            evaluate.validate_kitti(model.module, sradius_mode=args.sradius_mode)


def test(args):
    DEQFlow = get_model(args)
    model = nn.DataParallel(DEQFlow(args), device_ids=args.gpus)
    print("Parameter Count: %.3f M" % count_parameters(model))
    
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.eval()
    
    for test_dataset in args.test_set:
        if test_dataset == 'sintel':
            evaluate.create_sintel_submission(model.module, output_path=args.output_path,
                    fixed_point_reuse=args.fixed_point_reuse, warm_start=args.warm_start)
        elif test_dataset == 'kitti':
            evaluate.create_kitti_submission(model.module, output_path=args.output_path)


def visualize(args):
    DEQFlow = get_model(args)
    model = nn.DataParallel(DEQFlow(args), device_ids=args.gpus)
    print("Parameter Count: %.3f M" % count_parameters(model))
    
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.eval()
    
    for viz_dataset in args.viz_set:
        for split in args.viz_split:
            if viz_dataset == 'sintel':
                viz.sintel_visualization(model.module, split=split, output_path=args.output_path,
                        fixed_point_reuse=args.fixed_point_reuse, warm_start=args.warm_start)
            elif viz_dataset == 'kitti':
                viz.kitti_visualization(model.module, split=split, output_path=args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help="Enable Eval mode.")
    parser.add_argument('--test', action='store_true', help="Enable Test mode.")
    parser.add_argument('--viz', action='store_true', help="Enable Viz mode.")
    parser.add_argument('--fixed_point_reuse', action='store_true', help="Enable fixed point reuse.")
    parser.add_argument('--warm_start', action='store_true', help="Enable warm start.")
    
    parser.add_argument('--name', default='deq-flow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output_path', help="output path for evaluation")
    
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--large', action='store_true', help='use large model')
    parser.add_argument('--huge', action='store_true', help='use huge model')
    parser.add_argument('--gigantic', action='store_true', help='use gigantic model')
    
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--test_set', type=str, nargs='+')
    parser.add_argument('--viz_set', type=str, nargs='+')
    parser.add_argument('--viz_split', type=str, nargs='+', default=['test'])

    parser.add_argument('--eval_interval', type=int, default=5000, help="evaluation interval")
    parser.add_argument('--save_interval', type=int, default=20000, help="saving interval")
    parser.add_argument('--time_interval', type=int, default=500, help="timing interval")
    parser.add_argument('--resume_iter', type=int, default=-1, help="resume from the given iterations")

    parser.add_argument('--gma', action='store_true', help='use gma')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--schedule', type=str, default="onecycle", help="learning rate schedule")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--vdropout', type=float, default=0.0, help="variational dropout added to BasicMotionEncoder for DEQs")
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--wnorm', action='store_true', help="use weight normalization")
    parser.add_argument('--f_solver', default='anderson', type=str, choices=['anderson', 'broyden'],
                        help='forward solver to use (only anderson and broyden supported now)')
    parser.add_argument('--b_solver', default='broyden', type=str, choices=['anderson', 'broyden'],
                        help='backward solver to use')
    parser.add_argument('--f_thres', type=int, default=36, help='forward pass solver threshold')
    parser.add_argument('--b_thres', type=int, default=36, help='backward pass solver threshold')
    parser.add_argument('--stop_mode', type=str, default="abs", help="fixed-point convergence stop mode")
    parser.add_argument('--eval_factor', type=float, default=1.5, help="factor to scale up the f_thres at test for better precision.")

    parser.add_argument('--sliced_core', action='store_true', help="use the sliced core implementation.")
    parser.add_argument('--ift', action='store_true', help="use implicit differentiation.")
    parser.add_argument('--safe_ift', action='store_true', help="use a safer function for IFT to avoid segment fault.")
    parser.add_argument('--n_losses', type=int, default=1, help="number of loss terms (uniform spaced, 1 + fixed point correction).")
    parser.add_argument('--indexing', type=int, nargs='+', default=[], help="indexing for fixed point correction.")
    parser.add_argument('--sup_all', action='store_true', help="supervise all the trajectories by Phantom Grad.")
    parser.add_argument('--phantom_grad', type=int, nargs='+', default=[1], help="steps of Phantom Grad")
    parser.add_argument('--tau', type=float, default=1.0, help="damping factor for unrolled Phantom Grad")

    parser.add_argument('--sradius_mode', action='store_true', help="monitor spectral radius during validation")

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.eval:
        val(args)
    elif args.test:
        test(args)
    elif args.viz:
        visualize(args)
    else:
        train(args)
