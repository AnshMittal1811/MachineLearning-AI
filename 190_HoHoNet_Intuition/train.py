import os
import argparse
import importlib
from tqdm import tqdm, trange
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.config import config, update_config, infer_exp_id
from lib import dataset


def train_loop(net, loader, optimizer):
    net.train()
    if config.training.fix_encoder_bn:
        apply_fn_based_on_key(net.encoder, ['bn'], lambda m: m.eval())
    epoch_losses = Counter()
    for iit, batch in tqdm(enumerate(loader, 1), position=1, total=len(loader)):
        # Move data to the given computation device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        # feed forward & compute losses
        losses = net.compute_losses(batch)
        if len(losses) == 0:
            continue

        # backprop
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Log
        BS = len(batch['x'])
        epoch_losses['N'] += BS
        for k, v in losses.items():
            if torch.is_tensor(v):
                epoch_losses[k] += BS * v.item()
            else:
                epoch_losses[k] += BS * v

    # Statistic over the epoch
    N = epoch_losses.pop('N')
    for k, v in epoch_losses.items():
        epoch_losses[k] = v / N

    return epoch_losses


def valid_loop(net, loader):
    net.eval()
    epoch_losses = Counter()
    with torch.no_grad():
        for iit, batch in tqdm(enumerate(loader, 1), position=1, total=len(loader)):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # feed forward & compute losses
            losses = net.compute_losses(batch)

            # Log
            for k, v in losses.items():
                if torch.is_tensor(v):
                    epoch_losses[k] += float(v.item()) / len(loader)
                else:
                    epoch_losses[k] += v / len(loader)

    return epoch_losses


def apply_fn_based_on_key(net, key_lst, fn):
    for name, m in net.named_modules():
        if any(k in name for k in key_lst):
            fn(m)


def group_parameters(net, wd_group_mode):
    wd = []
    nowd = []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if wd_group_mode == 'bn and bias':
            if 'bn' in name or 'bias' in name:
                nowd.append(p)
            else:
                wd.append(p)
        elif wd_group_mode == 'encoder decoder':
            if 'feature_extractor' in name:
                nowd.append(p)
            else:
                wd.append(p)
    return [{'params': wd}, {'params': nowd, 'weight_decay': 0}]


if __name__ == '__main__':

    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    # Init global variable
    exp_id = infer_exp_id(args.cfg)
    exp_ckpt_root = os.path.join(config.ckpt_root, exp_id)
    os.makedirs(exp_ckpt_root, exist_ok=True)
    device = 'cuda' if config.cuda else 'cpu'
    if config.cuda and config.cuda_benchmark:
        torch.backends.cudnn.benchmark = True

    # Init dataset
    DatasetClass = getattr(dataset, config.dataset.name)
    config.dataset.train_kwargs.update(config.dataset.common_kwargs)
    config.dataset.valid_kwargs.update(config.dataset.common_kwargs)
    train_dataset = DatasetClass(**config.dataset.train_kwargs)
    valid_dataset = DatasetClass(**config.dataset.valid_kwargs)
    train_loader = DataLoader(train_dataset, config.training.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=config.num_workers,
                              pin_memory=config.cuda,
                              worker_init_fn=lambda x: np.random.seed())
    valid_loader = DataLoader(valid_dataset, 1,
                              num_workers=config.num_workers,
                              pin_memory=config.cuda)

    # Init network
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs).to(device)
    if config.training.fix_encoder_bn:
        apply_fn_based_on_key(net.encoder, ['bn'], lambda m: m.requires_grad_(False))

    # Init optimizer
    if config.training.optim == 'Adam':
        optimizer = torch.optim.Adam(
            group_parameters(net, config.training.wd_group_mode),
            lr=config.training.optim_lr, weight_decay=config.training.weight_decay)
    elif config.training.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            group_parameters(net, config.training.wd_group_mode),
            lr=config.training.optim_lr, weight_decay=config.training.weight_decay)
    elif config.training.optim == 'SGD':
        optimizer = torch.optim.SGD(
            group_parameters(net, config.training.wd_group_mode), momentum=0.9,
            lr=config.training.optim_lr, weight_decay=config.training.weight_decay)

    if config.training.optim_poly_gamma > 0:
        def lr_poly_rate(epoch):
            return (1 - epoch / config.training.epoch) ** config.training.optim_poly_gamma
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_poly_rate)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(p * config.training.epoch) for p in config.training.optim_milestons],
            gamma=config.training.optim_gamma)

    # Start training
    for iep in trange(1, config.training.epoch + 1, position=0):

        # Train phase
        epoch_losses = train_loop(net, train_loader, optimizer)
        scheduler.step()
        print(f'EP[{iep}/{config.training.epoch}] train:  ' +
              ' \ '.join([f'{k} {v:.3f}' for k, v in epoch_losses.items()]))

        # Periodically save model
        if iep % config.training.save_every == 0:
            torch.save(net.state_dict(), os.path.join(exp_ckpt_root, f'ep{iep}.pth'))
            print('Model saved')

        # Valid phase
        epoch_losses = valid_loop(net, valid_loader)
        print(f'EP[{iep}/{config.training.epoch}] valid:  ' +
              ' \ '.join([f'{k} {v:.3f}' for k, v in epoch_losses.items()]))

