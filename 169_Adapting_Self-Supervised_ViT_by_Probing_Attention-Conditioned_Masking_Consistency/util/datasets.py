# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from util.image_folder import ImageFolder
import os
import copy
import PIL
import numpy as np

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from RandAugment import RandAugment
from collections import Counter
import torch

# finetune_transforms is used when the model needs images with multiple augmentations when
# training with multiple objectives
def get_dataset(args, domains, transform_train, transform_val, finetune_transforms=None, eval_domains=None):
    if eval_domains is None:
        # Same data is used for training and evaluation
        eval_domains = domains

    dsets_train = [ImageFolder(os.path.join(domains[i]),
                            transform_train,
                            finetune_transform=finetune_transforms,
                            return_id=False, path_prefix=args.data_path) for i in range(len(domains))]
    dsets_val = [ImageFolder(os.path.join(eval_domains[i]),
                            transform_val,
                            finetune_transform=None,
                            return_id=False, path_prefix=args.data_path) for i in range(len(eval_domains))]
    dataset_train = torch.utils.data.ConcatDataset(dsets_train)
    dataset_val = torch.utils.data.ConcatDataset(dsets_val)
        
    return dataset_train, dataset_val


def get_dataloaders_for_domains(args, misc, domains, train_transforms, val_transforms=None, finetune_transforms=None, plabels=None, is_target=False, eval_domains=None):
    dataset_train, dataset_val = get_dataset(
        args, domains, train_transforms, val_transforms, finetune_transforms, eval_domains=eval_domains)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        print("Sampler_val = %s" % str(sampler_val))
    else:
        train_idx = np.arange(len(dataset_train))        
        sampler_train = torch.utils.data.SubsetRandomSampler(train_idx)
        sampler_val = torch.utils.data.SubsetRandomSampler(np.arange(len(dataset_val)))

    return get_train_val_loaders(dataset_train, sampler_train, dataset_val, sampler_val, args)


def get_loaders(args, misc, source_domains, train_transforms, val_transforms=None, target_domains=None, finetune_transforms=None, plabels=None,
    source_eval_domains=None, target_eval_domains=None):
    source_loaders = get_dataloaders_for_domains(args, misc, source_domains, train_transforms, val_transforms=val_transforms,
                                                 finetune_transforms=finetune_transforms, 
                                                 eval_domains=source_eval_domains)
    if target_domains is None:
        target_loaders = None
    else:
        target_loaders = get_dataloaders_for_domains(args, misc, target_domains, train_transforms, val_transforms=val_transforms,
                                                     finetune_transforms=finetune_transforms, plabels=plabels, is_target=True,
                                                     eval_domains=target_eval_domains)
    return source_loaders, target_loaders


def get_train_val_loaders(dataset_train, sampler_train, dataset_val, sampler_val, args):
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    return (data_loader_train, len(dataset_train)), (data_loader_val, len(dataset_val))

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    normalize_transform = transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # train transform
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(args.rand_augs, args.rand_aug_severity),
            transforms.ToTensor(),
            normalize_transform
        ])
        return transform

    # eval transforms (no center cropping)
    return transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        normalize_transform
    ])