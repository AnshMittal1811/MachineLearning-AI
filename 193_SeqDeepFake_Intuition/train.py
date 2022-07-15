import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,6,7'
from pathlib import Path
import sys
import argparse
import torch
import numpy as np
import random
import logging
from types import MethodType
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datasets.dataset import SeqDeepFakeDataset
from tools.utils import AverageMeter, NestedTensor

from tools.env import init_dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.configuration import Config
from models import SeqFakeFormer
import math


def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, acc_fixed, acc_adaptive):
        self.info('{set}-{idx:d} epoch | acc_fixed:{acc_fixed:.4f}% | acc_adaptive:{acc_adaptive:.4f}%'.format(
            set=set,
            idx=idx,
            acc_fixed=acc_fixed,
            acc_adaptive=acc_adaptive
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)


def preset_model(args, cfg, model, logger, sum_steps):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.lr_backbone,
        },
    ]         

    optimizer = torch.optim.AdamW(
        param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)
        
    if cfg.warmup:
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / cfg.warmup_epochs if epoch < cfg.warmup_epochs else 0.1**len([m for m in cfg.lr_milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        start_epoch = checkpoint['epoch']
        
        if args.log:
            logger.info(f'Loading model from {args.resume}...')
            logger.info(f'start_epoch: {start_epoch}...')

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(args.gpu)

        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        if args.log:
            logger.info('Create new model')

        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        start_epoch = 0

    return model, optimizer, start_epoch, scheduler

def read_csv(field, file):
    info = pd.read_csv(file)
    image_list = info[field[0]].tolist()
    score_list = info[field[1]].tolist()
    return image_list, score_list



def evalute(cfg, val_dataloader, model):
    # switch model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.PAD_token_id)
    criterion.eval()
    total = len(val_dataloader)

    with torch.no_grad():
        validation_loss = 0.0

        for steps, (images, masks, caps, cap_masks) in enumerate(tqdm(val_dataloader)):
            
            samples = NestedTensor(images, masks).to(0)
            caps = caps.cuda()
            cap_masks = cap_masks.cuda()

            input_caps = caps[:, :-1]
            pad_token_input_caps = cfg.PAD_token_id*torch.ones_like(input_caps)
            input_caps = torch.where(input_caps==cfg.EOS_token_id, pad_token_input_caps, input_caps)
            input_cap_masks = input_caps==cfg.PAD_token_id

            outputs = model(samples, input_caps, input_cap_masks)
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])


            validation_loss += loss.item()
            
        loss = validation_loss / total

    return loss


def create_caption_and_mask(cfg):
    caption_template = cfg.PAD_token_id*torch.ones((1, cfg.max_position_embeddings), dtype=torch.long).cuda()
    mask_template = torch.ones((1, cfg.max_position_embeddings), dtype=torch.bool).cuda()

    caption_template[:, 0] = cfg.SOS_token_id
    mask_template[:, 0] = False

    return caption_template, mask_template


def evalute_transformer(cfg, val_dataloader, model):
    # switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        running_corrects_fixed = 0.0
        epoch_size_fixed = 0.0 

        running_corrects_adaptive = 0.0
        epoch_size_adaptive = 0.0 

        for steps, (image, labels) in enumerate(tqdm(val_dataloader)):
            caption, cap_mask = create_caption_and_mask(cfg)
            image, labels = image.cuda(), labels.long().cuda()
            for i in range(cfg.max_position_embeddings - 1):
            
                predictions = model(image, caption, cap_mask)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)

                if predicted_id[0] == cfg.EOS_token_id:
                    caption = caption[:, 1:]
                    zero = torch.zeros_like(caption)
                    caption = torch.where(caption==cfg.PAD_token_id, zero, caption)
                    break

                caption[:, i+1] = predicted_id[0]
                cap_mask[:, i+1] = False
            
            if caption.shape[1] == 6:
                caption = caption[:, 1:]

            running_corrects_fixed += torch.sum(caption.cpu() == labels.data.cpu())
            epoch_size_fixed += image.size(0)*labels.shape[1]

            cmp_len = max(len(torch.where(labels[0]>0)[0]), len(torch.where(caption[0]>0)[0]))
            if cmp_len == 0:
                running_corrects_adaptive += 1
                cmp_len = 1
            else:
                running_corrects_adaptive += torch.sum(caption[:,:cmp_len].cpu() == labels[:,:cmp_len].data.cpu())
            epoch_size_adaptive += image.size(0)*cmp_len

    ACC_fixed =  running_corrects_fixed.double() / epoch_size_fixed
    ACC_adaptive =  running_corrects_adaptive.double() / epoch_size_adaptive

    return ACC_fixed, ACC_adaptive


def train(args, cfg, train_dataloader, train_sampler, val_dataloader, model, summary_writer, logger, log_dir):
    max_epochs = cfg.epochs
    max_iters = len(train_dataloader)
    sum_steps = max_epochs*max_iters

    model, optimizer, start_epoch, scheduler = preset_model(args, cfg, model, logger, sum_steps)

    criterion = torch.nn.CrossEntropyLoss(ignore_index = cfg.PAD_token_id).cuda(args.gpu)
    
    global_step = start_epoch*len(train_dataloader)
    if args.log:
        logger.info(f'global_step: {global_step}...')

    best_val_acc_fixed = 0        
    best_val_acc_adaptive = 0        

    for current_epoch in range(start_epoch, max_epochs):

        train_sampler.set_epoch(current_epoch)
        loss_logger = AverageMeter()
        # ----------
        #  Training
        # ----------
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if args.log:
            logger.info(f'############# Starting Epoch {current_epoch} | LR: {current_lr} #############')

        model.train()
        criterion.train()

        if args.log:
            train_dataloader = tqdm(train_dataloader, dynamic_ncols=True)
        for steps, (images, masks, caps, cap_masks) in enumerate(train_dataloader):
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            
            samples = NestedTensor(images, masks).to(args.gpu)
            caps = caps.cuda(args.gpu)
            cap_masks = cap_masks.cuda(args.gpu)

            input_caps = caps[:, :-1]
            pad_token_input_caps = cfg.PAD_token_id*torch.ones_like(input_caps)
            input_caps = torch.where(input_caps==cfg.EOS_token_id, pad_token_input_caps, input_caps)
            input_cap_masks = input_caps==cfg.PAD_token_id

            outputs = model(samples, input_caps, input_cap_masks)
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            if not math.isfinite(loss):
                print(f'Loss is {loss}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            optimizer.step()

            loss_logger.update(loss.item(), images.size(0))

            global_step+=1

            #============ tensorboard train log info ============#
            if args.log:
                lossinfo = {
                    'Train_Loss': loss.item(),                                                                                                  
                    'Train_Loss_avg': loss_logger.avg,                                                                                                  
                        } 
                for tag, value in lossinfo.items():
                    summary_writer.add_scalar(tag, value, global_step) 

                #============ print the train log info ============# 
                train_dataloader.set_description(
                    'lr: {lr:.8f} | loss: {loss:.8f} '.format(
                        loss=loss_logger.avg,
                        lr = current_lr
                    )
                )

        scheduler.step(current_epoch)
        #============ train model save ============#
        if args.model_save_epoch is not None:
            if (current_epoch % args.model_save_epoch == 0 and current_epoch != 0):
                if args.log:
                    model_save_path = os.path.join(log_dir, 'snapshots')    
                    mkdir(model_save_path) 
                    torch.save({
                        'epoch': current_epoch+1,
                        'state_dict': model.module.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        }, os.path.join(model_save_path, "model-{}.pt".format(current_epoch)))
        # ----------
        #  Validation
        # ----------
        if ((current_epoch+1) % args.val_epoch == 0):
            if args.log:
                model_save_path = os.path.join(log_dir, 'snapshots')    
                mkdir(model_save_path) 
                ACC_fixed, ACC_adaptive = evalute_transformer(cfg, val_dataloader, model.module)

                #============ print the val log info ============#
                logger.epochInfo('Validation', current_epoch, 100*ACC_fixed, 100*ACC_adaptive)
                #============ tensorboard val log info ============#
                valinfo = {
                    'Val_AUC_fixed': 100*ACC_fixed,               
                    'Val_AUC_adaptive': 100*ACC_adaptive,               
                        } 
                for tag, value in valinfo.items():
                    summary_writer.add_scalar(tag, value, current_epoch)
                
                if ACC_fixed >= best_val_acc_fixed:
                    best_val_acc_fixed = ACC_fixed
                    torch.save({
                        'best_val_acc_fixed': best_val_acc_fixed,
                        'best_state_dict_fixed': model.module.state_dict(),
                        }, os.path.join(model_save_path, "best_model_fixed.pt"))

                if ACC_adaptive >= best_val_acc_adaptive:
                    best_val_acc_adaptive = ACC_adaptive
                    torch.save({
                        'best_val_acc_adaptive': best_val_acc_adaptive,
                        'best_state_dict_adaptive': model.module.state_dict(),
                        }, os.path.join(model_save_path, "best_model_adaptive.pt"))


def main_worker(gpu, args, cfg):
    if gpu is not None:
        args.gpu = gpu

    init_dist(args)
    
    log_dir = os.path.join(args.results_dir, cfg.backbone, args.dataset_name, args.log_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log.txt')
    logger = setlogger(log_file)
    
    if args.log:
        summary_writer = SummaryWriter(log_dir)
    else:
        summary_writer = None
    
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(cfg.__dict__)
        logger.info('******************************')

    # model
    model = SeqFakeFormer.build_model(cfg)

    # TODO: check its performance
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    batch_size = cfg.batch_size
    
    train_dataset = SeqDeepFakeDataset(
        cfg=cfg,
        mode="train",
        data_root=args.data_dir,
        dataset_name=args.dataset_name
    )
    val_dataset = SeqDeepFakeDataset(
        cfg=cfg,
        mode="val",
        data_root=args.data_dir,
        dataset_name=args.dataset_name
    )
    if args.log:
        print('train:',len(train_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=4, sampler=train_sampler)
        
    if args.log:
        print('val:',len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    train(args, cfg, train_dataloader, train_sampler, val_dataloader, model, summary_writer, logger, log_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--cfg', type=str, default=None, help='path of config json file')
    arg('--results_dir', type=str, default='results')
    arg('--data_dir', type=str, default=None)
    arg('--dataset_name', type=str, default=None)
    arg('--resume', type=str, default=None)
    arg('--log_name', '-l', type=str)

    arg('--model_save_epoch', type=int, default=None)
    arg('--val_epoch', type=int, default=1)
    arg('--manual_seed', type=int, default=777)

    arg('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    arg('--world_size', default=1, type=int,
                        help='world size for distributed training')
    arg('--dist-url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    arg('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    arg('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')

    args = parser.parse_args()
    set_random_seed(args.manual_seed)
    cfg = Config(args.cfg)

    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, cfg))
