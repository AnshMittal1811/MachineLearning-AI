# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .vid import build_vid, build_det


import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    # def get_img_info(self, idx):
    #     dataset_idx, sample_idx = self.get_idxs(idx)
    #     return self.datasets[dataset_idx].get_img_info(sample_idx)


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


dataset_factory = {
'coco': build_coco,
"VID_train_15frames" : build_vid,
"DET_train_30classes": build_det,
"VID_val_videos" : build_vid
}





def build_dataset(image_set, cfg):
    if image_set == 'val':
        return dataset_factory[cfg.DATASET.val_dataset](image_set, cfg)
    if isinstance(cfg.DATASET.dataset_file, str):
        if cfg.DATASET.dataset_file == 'coco':
            return build_coco(image_set, cfg)
        if cfg.DATASET.dataset_file == 'coco_panoptic':
            # to avoid making panopticapi required for coco
            from .coco_panoptic import build as build_coco_panoptic
            return build_coco_panoptic(image_set, cfg)
        if cfg.DATASET.dataset_file == 'vid':
            return build_vid(image_set, cfg)
        raise ValueError(f'dataset {cfg.DATASET.dataset_file} not supported')
    elif isinstance(cfg.DATASET.dataset_file, list):
        if len(cfg.DATASET.dataset_file) > 1:
            datasets = []
            for dataset_name in cfg.DATASET.dataset_file:
                datasets.append(dataset_factory[dataset_name](image_set, cfg))

            dataset = ConcatDataset(datasets)
            return dataset
        else:
            dataset_name = cfg.DATASET.dataset_file[0]
            dataset = dataset_factory[dataset_name](image_set, cfg)
            return dataset


