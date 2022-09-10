import sys
import os
import os.path as osp

import torch
import util.misc as utils
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets.samplers as samplers
# from datasets import build_dataset, get_coco_api_from_dataset
from datasets import ConcatDataset
from datasets.vid_multi import build_vidmulti
from src.models.model_builder import build_model
from src.configs.default import _C as cfg
from src.engine.engine import evaluate, train_one_epoch
# from torch.utils.tensorboard import SummaryWriter
# from datasets.coco_hm import COCOHM
from util.misc import save_config, is_main_process


def main(cfg=cfg):

    parser = argparse.ArgumentParser(description="PyTorch Video Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="./experiments/PTSEFormer_r50_8gpus.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    # print(is_main_process())
    if (not os.path.exists(cfg.TRAIN.output_dir)) and is_main_process():
        try:
            os.mkdir(cfg.TRAIN.output_dir)
        except:
            pass
    print(cfg.TRAIN.output_dir)
    output_config_path = os.path.join(cfg.TRAIN.output_dir, 'config.yml')


    save_config(cfg, output_config_path)

    # tb_writer = SummaryWriter(os.path.join(cfg.TRAIN.output_dir, "tb_log"))
    tb_writer = None

    utils.init_distributed_mode(cfg.TRAIN)
    device = torch.device(cfg.TRAIN.device)
    seed = cfg.TRAIN.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    model, criterion, postprocessors = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    print(model_without_ddp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train_list = []
    for split_name in cfg.DATASET.dataset_file:
        dataset_train_list.append(build_vidmulti(image_set='train', cfg=cfg, split_name=split_name))
    # vidmulti_train = build_vidmulti(image_set='train', cfg=cfg, split_name='VID_train_15frames')
    # detmulti_train = build_vidmulti(image_set='train', cfg=cfg, split_name='DET_train_30classes')
    dataset_train = ConcatDataset(dataset_train_list)

    # dataset_train =build_vidmulti(image_set='train', cfg=cfg, split_name='VID_train_3frames')

    dataset_val = build_vidmulti(image_set='val', cfg=cfg, split_name=cfg.DATASET.val_dataset)

    if cfg.TRAIN.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.TRAIN.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn,
                                   num_workers=cfg.TRAIN.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, cfg.TEST.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.TRAIN.num_workers,
                                 pin_memory=True)


    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, cfg.TRAIN.lr_backbone_names) and not match_name_keywords(n, cfg.TRAIN.lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.TRAIN.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.lr_backbone_names) and p.requires_grad],
            "lr": cfg.TRAIN.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.TRAIN.lr * cfg.TRAIN.lr_linear_proj_mult,
        }
    ]




    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.lr,
                                  weight_decay=cfg.TRAIN.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.lr_drop)
    if cfg.TRAIN.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.TRAIN.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    # base_ds = get_coco_api_from_dataset(dataset_val)
    base_ds = dataset_val.coco

    output_dir = Path(cfg.TRAIN.output_dir)

    start_epoch = cfg.TRAIN.start_epoch
    ap50_best = 0.0
    if cfg.TRAIN.resume or (cfg.TRAIN.resume_default and list(filter(lambda x: x.endswith(".pth"), os.listdir(cfg.TRAIN.output_dir)))):
        if cfg.TRAIN.resume:
            print("resume: ", cfg.TRAIN.resume)
            checkpoint = torch.load(cfg.TRAIN.resume, map_location='cpu')
            resume_dir = cfg.TRAIN.resume

        else:
            resume_path = sorted(filter(lambda x: x.endswith(".pth"), os.listdir(cfg.TRAIN.output_dir)))
            resume_dir = osp.join(cfg.TRAIN.output_dir, resume_path[-1])
            if osp.exists((osp.join(cfg.TRAIN.output_dir, 'checkpoint.pth'))):
                resume_dir = osp.join(cfg.TRAIN.output_dir, 'checkpoint.pth')
            checkpoint = torch.load(resume_dir, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        print("resume: ", resume_dir)
        print("loading all.")

        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            # args.override_resumed_lr_drop = True
            # if args.override_resumed_lr_drop:
            print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
            lr_scheduler.step_size = cfg.TRAIN.lr_drop
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            start_epoch = checkpoint['epoch'] + 1

        print(len(data_loader_val))
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.TRAIN.output_dir, log_every_dir=osp.join(output_dir, 'log_every.txt')
        )

        ap50_best = coco_evaluator.coco_eval['bbox'].stats.tolist()[1]

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.TRAIN.epochs):
        if cfg.TRAIN.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, cfg.TRAIN.clip_max_norm, tb_writer=tb_writer, log_every_dir=osp.join(output_dir, 'log_every.txt'))
        lr_scheduler.step()
        if cfg.TRAIN.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % cfg.TRAIN.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': cfg,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.TRAIN.output_dir, log_every_dir=osp.join(output_dir, 'log_every.txt')
        )

        ap50_best_tmp = coco_evaluator.coco_eval['bbox'].stats.tolist()[1]

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}


        if cfg.TRAIN.output_dir and utils.is_main_process():
            if ap50_best_tmp >= ap50_best:
                ap50_best = ap50_best_tmp
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': cfg,
                }, output_dir / f'checkpoint_best.pth')


            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    main()