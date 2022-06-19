# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from engine_pacmac_adapt import train_one_epoch_with_target
from engine_finetune import evaluate_and_log, evaluate
import models_vit
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc
from util.logging import update_log_stats_dict
import argparse
import distutils
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from util.datasets import build_transform, get_loaders
import timm
from timm.loss import LabelSmoothingCrossEntropy
assert timm.__version__ == "0.3.2"  # version check

import util.lr_decay as lrd

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--img_log_freq', default=100,
                        type=int, help='step frequency of logging reconstructed images')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument("--checkpoint_key", default="model", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--ckpt_save_freq', default=50,
                        type=int, help='epoch frequency of saving ckpts')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    # Augmentation parameters
    parser.add_argument('--rand_augs', type=int, default=0,
                        help='number of augmentations transforms to apply')
    parser.add_argument('--rand_aug_severity', type=float,
                        default=1, help='the severity of RandAug transforms')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--target_smoothing', type=float, default=0.1,
                        help='Target label smoothing (default: 0.1)')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--source_domains', nargs="+", required=True)
    parser.add_argument('--target_domains', nargs="+", required=True)

    # for monitoring accuracies on source/target domain
    parser.add_argument('--source_eval_domains', nargs="+",
                        default=None)
    parser.add_argument('--target_eval_domains', nargs="+", default=None)

    parser.add_argument('--target_eval_freq', default=1000, type=int)
    parser.add_argument('--source_eval_freq', default=1, type=int,
                        help='epoch frequency of evaluation on source val dataset')
    parser.add_argument('--attention_seeding',
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        default=True,
                        help='Use attention-seeded masking')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='confidence threshold for self-training on target')
    parser.add_argument('--lambda_unsup', type=float, default=0.1,
                        help='Unsupervised loss weight')
    parser.add_argument('--committee_size', type=int, default=2,
                        help='committee size')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    misc.log_arguments(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    pretrain_transform_train = build_transform(is_train=False, args=args)
    transform_val = build_transform(is_train=False, args=args)
    finetune_transform_train = build_transform(is_train=True, args=args)

    ((source_loader_train, _), (source_loader_val, source_val_len)), ((target_loader_train, _), (target_loader_val, target_val_len)) = \
        get_loaders(args, misc, args.source_domains, train_transforms=pretrain_transform_train,
                    val_transforms=transform_val, target_domains=args.target_domains, finetune_transforms=finetune_transform_train,
                    source_eval_domains=args.source_eval_domains, target_eval_domains=args.target_eval_domains)
    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    model = models_vit.__dict__[args.model](num_classes=args.nb_classes,
                                            drop_path_rate=args.drop_path,
                                            global_pool=args.global_pool)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    print(optimizer)

    loss_scaler = NativeScaler()

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    target_criterion = criterion
    
    print("criterion = %s" % str(criterion))
    print("target criterion = %s" % str(target_criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, strict=False, load_optim_from_ckpt=False)
    
    if args.eval:        
        test_stats = evaluate(source_loader_val, model, device, num_classes=args.nb_classes)
        target_test_stats = evaluate(target_loader_val, model, device, num_classes=args.nb_classes)
        print(f"Top-1 acc. of the network on the {source_val_len} source test images: {test_stats['acc1']:.1f}%")
        print(f"Top-1 acc. of the network on the {target_val_len} target test images: {target_test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            source_loader_train.sampler.set_epoch(epoch)
            target_loader_train.sampler.set_epoch(epoch)
        # Uses a zipped loader (all source samples may not be considered)
        train_stats = train_one_epoch_with_target(
            model, criterion, target_criterion, source_loader_train, target_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args)
        if args.output_dir and ((epoch + 1) % args.ckpt_save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {}
        update_log_stats_dict(log_stats, train_stats, epoch, 'source train')

        # evaluate on source images
        if (epoch + 1) % args.source_eval_freq == 0:
            evaluate_and_log(model, source_loader_val, device, source_val_len, epoch, log_writer, log_stats, 'source test', args)

        # evaluate on target images
        if (epoch + 1) % args.target_eval_freq == 0 or epoch + 1 == args.epochs:
            target_test_stats = evaluate_and_log(model, target_loader_val, device, target_val_len, epoch, log_writer, log_stats, 'target test', args)
            max_accuracy = max(max_accuracy, target_test_stats["acc1"])
            print(f'Max target accuracy: {max_accuracy:.2f}%')

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            if len(log_stats) > 0:
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
