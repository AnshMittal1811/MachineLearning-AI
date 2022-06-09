from __future__ import print_function, division
import sys

sys.path.append('..')
sys.path.append('../core')
sys.path.append('../datasets')

import argparse
import os
import cv2
import time
import numpy as np
import json
import glob
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from raft import RAFT
from llff import LLFF, LLFFTest

import projective_ops as pops

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from collections import OrderedDict
import subprocess


# import tracemalloc
# tracemalloc.start()


def sequence_loss_slant(slant_est, slant_gt, disp_gt, gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(slant_est)
    flow_loss = 0.0

    valid = torch.logical_and(disp_gt > 0.0, torch.logical_and(torch.abs(slant_gt[:, 0] < 0.00001),
                                                               torch.abs(slant_gt[:, 1] < 0.00001)))
    ht, wd = slant_gt.shape[-2:]
    # print(slant_est[-1].shape)
    ht_est, wd_est = slant_est[-1].shape[-2:]

    for i in range(n_predictions):
        slant_est[i] = F.interpolate(slant_est[i][:, 0, :, :, :], [ht, wd], mode='bilinear',
                                     align_corners=True) * ht_est / ht

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (slant_est[i] - slant_gt).abs()

        flow_loss += i_weight * (valid * i_loss).mean()
        flow_loss += .01 * i_weight * (i_loss).mean()

    assert (not torch.isnan(slant_est[-1]).any())
    assert (not torch.isinf(slant_est[-1]).any())

    return flow_loss * 10


def sequence_loss(disp_est, disp_gt,
                  loss_type='depth_gradual',
                  depthloss_threshold=100,
                  weight=None,
                  gradual_weight=None,
                  gamma=0.9,
                  disp_clamp=1e-4
                  ):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_est)
    flow_loss = 0.0

    valid = disp_gt > 0.0


    ht, wd = disp_gt.shape[-2:]

    for i in range(n_predictions):
        disp_est[i] = F.interpolate(disp_est[i], [ht, wd], mode='bilinear', align_corners=True)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        if loss_type == "disp":
            i_loss = (disp_est[i] - disp_gt).abs()
        elif loss_type == "depth":
            i_loss = (1.0 / disp_est[i].clamp(min=disp_clamp) - 1.0 / disp_gt.clamp(min=disp_clamp)).abs()
            i_loss = i_loss.clamp(max=depthloss_threshold) / 3.6e5
            # i_loss = i_loss / 3.6e5
        elif loss_type == "depth_softloss":
            i_loss = (1.0 / disp_est[i].clamp(min=disp_clamp) - 1.0 / disp_gt.clamp(min=disp_clamp)).abs()
            i_loss = depthloss_threshold * i_loss / (i_loss + depthloss_threshold) / 3.6e5
        elif loss_type == "depth_gradual":
            i_loss = (1.0 / disp_est[i].clamp(min=disp_clamp) - 1.0 / disp_gt.clamp(min=disp_clamp)).abs()
            i_loss = i_loss.clamp(max=depthloss_threshold) / 3.6e5
            i_loss = gradual_weight * i_loss + (1 - gradual_weight) * (disp_est[i] - disp_gt).abs()

        # if i == n_predictions - 1:
        #     print('a', (valid * i_loss * 3.6e5).mean().item())
        #     print('b', (i_loss * 3.6e5).view(-1)[valid.view(-1)].mean().item())

        if not weight is None:
            i_loss *= weight
        # flow_loss += i_weight * (valid * i_loss).mean()
        flow_loss += i_weight * (valid * i_loss).sum() / torch.sum(valid)
        # flow_loss += .01 * i_weight * (i_loss).mean()

    # epe = (1.0 / disp_est[-1].clamp(min=.001) - 1.0 / disp_gt).abs()
    # epe = epe.view(-1)[valid.view(-1)]

    epe = (1.0 / disp_est[-1].clamp(min=disp_clamp) - 1.0 / disp_gt.clamp(min=disp_clamp)).abs()
    epe = epe.view(-1)[valid.view(-1)]

    # print(loss_type)
    # print('num valid pixels: ', torch.sum(valid))
    # print('loss: ', flow_loss.item())
    # print('epe: ', epe.mean().item())

    assert (not torch.isnan(disp_est[-1]).any())
    assert (not torch.isinf(disp_est[-1]).any())

    # print(torch.max(epe), torch.min(epe), torch.mean(epe))

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 3).float().mean().item(),
        '3px': (epe < 10).float().mean().item(),
        '5px': (epe < 25).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=args.pct_start, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, output, SUM_FREQ):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.output = output
        self.SUM_FREQ = SUM_FREQ

    def _print_training_status(self):
        SUM_FREQ = self.SUM_FREQ
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)
        if not self.output is None:
            f = open(self.output, "a")
            f.write(f"{training_str + metrics_str}\n")
            f.close()

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        SUM_FREQ = self.SUM_FREQ
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
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
    params = {
        "corr_len": 2 if args.pooltype in ['maxmean', "meanvar"] else 1,
        "inference": 0,
    }
    for k in list(vars(args).keys()):
        params[k] = vars(args)[k]

    with open('dir.json') as f:
        d = json.load(f)
    d = d[args.setting]

    model = RAFT(**params)
    model = model.cuda()

    # if args.fix_mistake:
    #     max_step = 0
    #     for i in range(0, 100001, 5000):
    #         if os.path.exists('checkpoints/%d_%s.pth' % (i, args.name[:-2])):
    #             max_step = i
    #     print(max_step)
    #     tmp = torch.load('checkpoints/%d_%s.pth' % (max_step, args.name[:-2]))
    #     if list(tmp.keys())[0][:7] == "module.":
    #         model = nn.DataParallel(model)
    #     model.load_state_dict(tmp, strict=False)
    #     assert(0)
    # else:
    # since finetune
    assert args.restore_ckpt is not None

    if args.restore_ckpt is not None:
        tmp = torch.load(args.restore_ckpt)
        if list(tmp.keys())[0][:7] == "module.":
            model = nn.DataParallel(model)
        model.load_state_dict(tmp, strict=False)
    else:
        model = nn.DataParallel(model)

    optimizer, scheduler = fetch_optimizer(args, model)

    gpuargs = {'num_workers': 1, 'drop_last': True, 'shuffle': args.shuffle, 'pin_memory': True}

    datasetname = d["dataset"]
    dataset_args = {"num_frames": args.num_frames,
                    "crop_size": [args.crop_h, args.crop_w],
                    "resize": [args.resize_h, args.resize_w]}

    if datasetname == "DTU":
        dataset_args["pairs_provided"] = args.pairs_provided
        dataset_args["light_number"] = args.light_number
    elif datasetname == "Blended":
        dataset_args["scale"] = args.scale
        dataset_args["scaling"] = args.scaling
        dataset_args["image_aug"] = args.image_aug
    elif datasetname == "LLFF":
        dataset_args["data_augmentation"] = args.data_augmentation

        total_num_views = len(sorted(glob.glob(os.path.join(d["testing_dir"], args.single_scan, "DTU_format", "images", "*.jpg"))))
        indicies = np.arange(total_num_views)
        print(indicies)
        dataset_args["source_views"] = list(indicies[np.mod(np.arange(len(indicies), dtype=int), 5) != 2])

    dataset_path = os.path.join(d["testing_dir"])

    if args.single_scan != "":
        dataset_path = os.path.join(dataset_path, args.single_scan)

    print('dataset_path:', dataset_path)

    train_dataset = eval(datasetname)(dataset_path, **dataset_args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **gpuargs)

    model.eval()

    total_steps = 0
    scaler = GradScaler(enabled=True)
    logger = Logger(model, scheduler, args.outputfile, args.SUM_FREQ)

    VAL_FREQ = args.VAL_FREQ

    should_keep_training = True

    tic = None
    total_time = 0

    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):

            optimizer.zero_grad()
            if datasetname == "DTU":
                images, depths, poses, intrinsics = data_blob
            elif datasetname == "Blended":
                images, depths, poses, intrinsics, scene_id, indices, scale, SD, ED, ND = data_blob
            elif datasetname == "LLFF":
                images, depths, poses, intrinsics = data_blob

            graph = OrderedDict()
            graph[0] = list(range(1, args.num_frames))
            depths = depths.cuda()
            depths = depths[:, [0]]
            disp_gt = torch.where(depths > 0, 1.0 / depths, torch.zeros_like(depths))

            if args.slant:
                disp_est, slant_est = model(images.cuda(), poses.cuda(), intrinsics.cuda(), graph)
            else:
                if not args.invariance:
                    disp_est = model(images.cuda(), poses.cuda(), intrinsics.cuda(), graph)
                else:
                    disp_est, invariance_loss = model(images.cuda(), poses.cuda(), intrinsics.cuda(), graph)

            weight = None

            if args.Euclidean:
                ht, wd = disp_gt.shape[-2:]
                x_grid, y_grid = torch.meshgrid(torch.arange(ht), torch.arange(wd))
                x_grid = x_grid.cuda().float()
                y_grid = y_grid.cuda().float()
                x_grid = x_grid.view(1, 1, ht, wd).repeat(args.batch_size, 1, 1, 1)
                y_grid = y_grid.view(1, 1, ht, wd).repeat(args.batch_size, 1, 1, 1)
                x_grid -= intrinsics.cuda()[:, 0:1, 1:2, 2:3]
                y_grid -= intrinsics.cuda()[:, 0:1, 0:1, 2:3]
                x_grid /= intrinsics.cuda()[:, 0:1, 1:2, 1:2]
                y_grid /= intrinsics.cuda()[:, 0:1, 0:1, 0:1]
                weight = (x_grid ** 2 + y_grid ** 2 + 1) ** 0.5

            loss_type = "disp"
            if total_steps > args.disp_runup and (args.loss_type == 'depth'):
                loss_type = "depth"
            if total_steps > args.disp_runup and (args.loss_type == 'depth_softloss'):
                loss_type = "depth_softloss"
            if args.loss_type == 'depth_gradual':
                loss_type = "depth_gradual"
                gradual_weight = total_steps * 1.0 / args.num_steps
            else:
                gradual_weight = None

            loss, metrics = sequence_loss(disp_est, disp_gt,
                                          depthloss_threshold=args.depthloss_threshold,
                                          loss_type=loss_type,
                                          weight=weight,
                                          gradual_weight=gradual_weight)

            # print("loss", loss)
            # print("memory-forward", report())
            torch.set_printoptions(precision=15)

            # if True:
            #     loss.backward()
            # else:

            # print(scaler.get_scale())
            scaler.scale(loss).backward()
            # print(scaler.get_scale())
            # print("memory-backward", report())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            # print('scale:', scaler.get_scale()) if need debug

            logger.push(metrics)

            # state = model.state_dict()
            # if total_steps % VAL_FREQ == VAL_FREQ - 1 or total_steps == 1:
            #     PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
            #     torch.save(model.state_dict(), PATH)

            total_steps += 1

            # if total_steps == 1:
            #     should_keep_training = False
            #     break

            if not tic is None:
                total_time += time.time() - tic
                print(
                    f"time per step: {total_time / (total_steps - 1)}, expected: {total_time / (total_steps - 1) * args.num_steps / 24 / 3600} days")
            tic = time.time()

            # stats = torch.cuda.memory_stats()
            # peak_bytes_requirement = stats["allocated_bytes.all.peak"]
            # print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")
            # current, peak =  tracemalloc.get_traced_memory()
            # print(f"{current:0.2f}, {peak:0.2f}")

            # print("memory:", report())

            if total_steps > min(args.pause_steps, args.num_steps):
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # #================= not used =======================
    # parser.add_argument('--stage', help="determines which dataset to use for training")
    # parser.add_argument('--mode', default='stereo')
    # parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--clip', type=float, default=1.0)
    # parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--gamma', type=float, default=0.9, help='exponential weighting')
    # parser.add_argument('--add_noise', action='store_true')
    # parser.add_argument('--validation', type=str, nargs='+')
    # parser.add_argument('--debug', type=int, default=False)
    # #================= not used =======================

    ''' training args'''
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--pct_start', type=float, default=0.001)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--pause_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=int, default=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--SUM_FREQ', type=int, default=100)
    parser.add_argument('--VAL_FREQ', type=int, default=5000)
    parser.add_argument('--outputfile', type=str,
                        default=None)  # in case stdoutput is buffered (don't know how to disable buffer...)

    '''loss args'''
    parser.add_argument('--loss_type', type=str, default='depth_gradual',
                        choices=['disp', "depth", "depth_softloss", 'depth_gradual'])
    parser.add_argument('--depthloss_threshold', type=float, default=100)
    parser.add_argument('--disp_runup', type=int, default=10000)
    parser.add_argument('--Euclidean', type=int, default=True)

    '''dataset args'''
    parser.add_argument('--setting', type=str, default='DTU')
    parser.add_argument('--single_scan', type=str, default='')
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--light_number', type=int, default=-1)
    parser.add_argument('--crop_h', type=int, default=448)
    parser.add_argument('--crop_w', type=int, default=576)
    parser.add_argument('--resize_h', type=int, default=-1)
    parser.add_argument('--resize_w', type=int, default=-1)
    parser.add_argument('--pairs_provided', type=int, default=0)
    parser.add_argument('--scaling', type=str, default="median")
    parser.add_argument('--image_aug', type=int, default=False)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--data_augmentation', type=int, default=True)

    '''model args'''
    '''prediction range'''
    parser.add_argument('--Dnear', type=float, default=.0025)
    parser.add_argument('--DD', type=int, default=128)
    parser.add_argument('--Dfar', type=float, default=.0)
    parser.add_argument('--DD_fine', type=int, default=320)
    # parser.add_argument('--len_dyna', type=int, default=44)
    '''layer and kernel size'''
    parser.add_argument('--kernel_z', type=int, default=3)
    parser.add_argument('--kernel_r', type=int, default=3)
    parser.add_argument('--kernel_q', type=int, default=3)
    parser.add_argument('--kernel_corr', type=int, default=3)
    parser.add_argument('--dim0_corr', type=int, default=128)
    parser.add_argument('--dim1_corr', type=int, default=128)
    parser.add_argument('--dim_net', type=int, default=128)
    parser.add_argument('--dim_inp', type=int, default=128)
    parser.add_argument('--dim0_delta', type=int, default=256)
    parser.add_argument('--kernel0_delta', type=int, default=3)
    parser.add_argument('--kernel1_delta', type=int, default=3)
    parser.add_argument('--dim0_upmask', type=int, default=256)
    parser.add_argument('--kernel_upmask', type=int, default=3)
    parser.add_argument('--num_levels', type=int, default=5)
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--s_disp_enc', type=int, default=7)
    parser.add_argument('--dim_fmap', type=int, default=128)
    '''variants'''
    parser.add_argument('--num_iters', type=int, default=16)
    parser.add_argument('--HR', type=int, default=False)
    parser.add_argument('--HRv2', type=int, default=False)
    parser.add_argument('--cascade', type=int, default=False)
    parser.add_argument('--cascade_v2', type=int, default=False)
    parser.add_argument('--num_iters1', type=int, default=8)
    parser.add_argument('--num_iters2', type=int, default=5)
    parser.add_argument('--slant', type=int, default=False)
    parser.add_argument('--invariance', type=int, default=False)
    parser.add_argument('--pooltype', type=str, default="maxmean")
    parser.add_argument('--no_upsample', type=int, default=False)
    parser.add_argument('--merge', type=int, default=False)
    parser.add_argument('--visibility', type=int, default=False)
    parser.add_argument('--visibility_v2', type=int, default=False)
    # parser.add_argument('--merge_permute', type=int, default=0)

    # parser.add_argument('--fix_mistake', type=int, default=False)

    args = parser.parse_args()
    args.len_dyna = (2 * args.radius + 1) * 2 ** (args.num_levels - 1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
