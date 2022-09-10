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
        default="./experiments/PTSEFormer_r101_8gpus.yaml",
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

    dataset_val = build_vidmulti(image_set='val', cfg=cfg, split_name=cfg.DATASET.val_dataset)

    if cfg.TRAIN.distributed:
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, cfg.TEST.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.TRAIN.num_workers,
                                 pin_memory=True)


    if cfg.TRAIN.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.TRAIN.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    # base_ds = get_coco_api_from_dataset(dataset_val)
    base_ds = dataset_val.coco

    output_dir = Path(cfg.TRAIN.output_dir)

    print("resume: ", cfg.TRAIN.resume)
    checkpoint = torch.load(cfg.TRAIN.resume, map_location='cpu')

    model_without_ddp.load_state_dict(checkpoint['model'])
    print("loading all.")

    print(len(data_loader_val))
    test_stats, coco_evaluator = evaluate(
        model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.TRAIN.output_dir, log_every_dir=osp.join(output_dir, 'log_every.txt')
    )

if __name__ == '__main__':
    main()