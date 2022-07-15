import os
import argparse
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from model import Encoder, Decoder
from dataset import PanoDataset
from utils import group_weight, adjust_learning_rate, StatisticDict, pmap_x


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--id', required=True,
                    help='experiment id to name checkpoints')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')
# Dataset related arguments
parser.add_argument('--root_dir_train', default='data/train',
                    help='root directory for training data')
parser.add_argument('--root_dir_valid', default='data/valid',
                    help='root directory for validation data')
parser.add_argument('--input_cat', default=['img', 'line'], nargs='+',
                    help='input channels subdirectories')
parser.add_argument('--input_channels', default=6, type=int,
                    help='numbers of input channels')
parser.add_argument('--no_flip', action='store_true',
                    help='disable left-right flip augmentation')
parser.add_argument('--no_rotate', action='store_true',
                    help='disable horizontal rotate augmentation')
parser.add_argument('--no_gamma', action='store_true',
                    help='disable gamma augmentation')
parser.add_argument('--noise', action='store_true',
                    help='enable noise augmentation')
parser.add_argument('--contrast', action='store_true',
                    help='enable contrast augmentation')
parser.add_argument('--num_workers', default=6, type=int,
                    help='numbers of workers for dataloaders')
# optimization related arguments
parser.add_argument('--batch_size_train', default=2, type=int,
                    help='training mini-batch size')
parser.add_argument('--batch_size_valid', default=2, type=int,
                    help='validation mini-batch size')
parser.add_argument('--epochs', default=50, type=int,
                    help='epochs to train')
parser.add_argument('--optim', default='Adam',
                    help='optimizer to use. only support SGD and Adam')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--lr_pow', default=0, type=float,
                    help='power in poly to drop LR')
parser.add_argument('--warmup_lr', default=1e-6, type=float,
                    help='starting learning rate for warm up')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='numbers of warmup epochs')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='factor for L2 regularization')
parser.add_argument('--cormap_smooth', default=0, type=float,
                    help='cor probability smooth constraint')
# Misc arguments
parser.add_argument('--no_cuda', action='store_true',
                    help='disable cuda')
parser.add_argument('--seed', default=277, type=int,
                    help='manual seed')
parser.add_argument('--disp_iter', type=int, default=20,
                    help='iterations frequency to display')
parser.add_argument('--save_every', type=int, default=5,
                    help='epochs frequency to save state_dict')
args = parser.parse_args()
device = torch.device('cpu' if args.no_cuda else 'cuda')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)


# Create dataloader
dataset_train = PanoDataset(root_dir=args.root_dir_train,
                            cat_list=[*args.input_cat, 'edge', 'cor'],
                            flip=not args.no_flip, rotate=not args.no_rotate,
                            gamma=not args.no_gamma, noise=args.noise,
                            contrast=args.contrast)
dataset_valid = PanoDataset(root_dir=args.root_dir_valid,
                            cat_list=[*args.input_cat, 'edge', 'cor'],
                            flip=False, rotate=False,
                            gamma=False, noise=False,
                            contrast=False)
loader_train = DataLoader(dataset_train, args.batch_size_train,
                          shuffle=True, drop_last=True,
                          num_workers=args.num_workers,
                          pin_memory=not args.no_cuda)
loader_valid = DataLoader(dataset_valid, args.batch_size_valid,
                          shuffle=False, drop_last=False,
                          num_workers=args.num_workers,
                          pin_memory=not args.no_cuda)


# Create model
encoder = Encoder(args.input_channels).to(device)
edg_decoder = Decoder(skip_num=2, out_planes=3).to(device)
cor_decoder = Decoder(skip_num=3, out_planes=1).to(device)


# Create optimizer
if args.optim == 'SGD':
    optimizer = optim.SGD(
        [*group_weight(encoder), *group_weight(edg_decoder), *group_weight(cor_decoder)],
        lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(
        [*group_weight(encoder), *group_weight(edg_decoder), *group_weight(cor_decoder)],
        lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
else:
    raise NotImplementedError()


# Init variable
args.warmup_iters = args.warmup_epochs * len(loader_train)
args.max_iters = args.epochs * len(loader_train)
args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
args.cur_iter = 0
train_losses = StatisticDict(winsz=100)
criti = nn.BCEWithLogitsLoss(reduction='none')

print(args)
print('%d iters per epoch for train' % len(loader_train))
print('%d iters per epoch for valid' % len(loader_valid))
print(' start training '.center(80, '='))


# Start training
for ith_epoch in range(1, args.epochs + 1):
    for ith_batch, datas in enumerate(loader_train):
        # Set learning rate
        adjust_learning_rate(optimizer, args)
        args.cur_iter += 1

        # Prepare data
        x = torch.cat([datas[i]
                      for i in range(len(args.input_cat))], dim=1).to(device)
        y_edg = datas[-2].to(device)
        y_cor = datas[-1].to(device)

        # Feedforward
        en_list = encoder(x)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])
        y_edg_ = edg_de_list[-1]
        y_cor_ = cor_de_list[-1]

        # Compute loss
        loss_edg = criti(y_edg_, y_edg)
        loss_edg[y_edg == 0.] *= 0.2
        loss_edg = loss_edg.mean()
        loss_cor = criti(y_cor_, y_cor)
        loss_cor[y_cor == 0.] *= 0.2
        loss_cor = loss_cor.mean()
        loss = loss_edg + loss_cor

        if args.cormap_smooth > 0:
            cormap = torch.sigmoid(y_cor_)
            LR = pmap_x(y_cor[..., :-1], y_cor[..., 1:])
            LR_ = pmap_x(cormap[..., :-1], cormap[..., 1:])
            UB = pmap_x(y_cor[..., :-1, :], y_cor[..., 1:, :])
            UB_ = pmap_x(cormap[..., :-1, :], cormap[..., 1:, :])

            LR_loss = (LR - LR_).abs().mean() * args.cormap_smooth
            UB_loss = (UB - UB_).abs().mean() * args.cormap_smooth
            loss += LR_loss + UB_loss

            train_losses.update('lr loss', LR_loss.item())
            train_losses.update('ub loss', UB_loss.item())

        # backprop
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(chain(
            encoder.parameters(), edg_decoder.parameters(), cor_decoder.parameters()),
            3.0, norm_type='inf')
        optimizer.step()

        # Statitical result
        train_losses.update('edg loss', loss_edg.item())
        train_losses.update('cor loss', loss_cor.item())
        if args.cur_iter % args.disp_iter == 0:
            print('iter %d (epoch %d) | lr %.6f | %s' % (
                args.cur_iter, ith_epoch, args.running_lr, train_losses),
                flush=True)

    # Dump model
    if ith_epoch % args.save_every == 0:
        torch.save(encoder.state_dict(),
                   os.path.join(args.ckpt, args.id, 'epoch_%d_encoder.pth' % ith_epoch))
        torch.save(edg_decoder.state_dict(),
                   os.path.join(args.ckpt, args.id, 'epoch_%d_edg_decoder.pth' % ith_epoch))
        torch.save(cor_decoder.state_dict(),
                   os.path.join(args.ckpt, args.id, 'epoch_%d_cor_decoder.pth' % ith_epoch))
        print('model saved')

    # Validate
    valid_losses = StatisticDict()
    for ith_batch, datas in enumerate(loader_valid):
        with torch.no_grad():
            # Prepare data
            x = torch.cat([datas[i]
                          for i in range(len(args.input_cat))], dim=1).to(device)
            y_edg = datas[-2].to(device)
            y_cor = datas[-1].to(device)

            # Feedforward
            en_list = encoder(x)
            edg_de_list = edg_decoder(en_list[::-1])
            cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])
            y_edg_ = edg_de_list[-1]
            y_cor_ = cor_de_list[-1]

            # Compute loss
            loss_edg = criti(y_edg_, y_edg)
            loss_edg[y_edg == 0.] *= 0.2
            loss_edg = loss_edg.mean()
            loss_cor = criti(y_cor_, y_cor)
            loss_cor[y_cor == 0.] *= 0.2
            loss_cor = loss_cor.mean()

            valid_losses.update('edg loss', loss_edg.item(), weight=x.size(0))
            valid_losses.update('cor loss', loss_cor.item(), weight=x.size(0))
    print('validation | epoch %d | %s' % (ith_epoch, valid_losses), flush=True)
