# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Submodule interface.
"""
import warnings
from pycocotools.coco import COCO
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import CocoDetection

from .coco import build as build_coco
from .vis import build as build_vis
from .coco_joint_vis import build_coco_joint_vis

def get_coco_api_from_dataset(dataset: Subset) -> COCO:
    """Return COCO class from PyTorch dataset for evaluation with COCO eval."""
    for _ in range(10):
        # if isinstance(dataset, CocoDetection):
        #     break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, CocoDetection):
        raise NotImplementedError

    return dataset.coco


def build_dataset(image_set, cfg):
    if cfg.DATASETS.TYPE == 'coco':
        return build_coco(image_set, cfg)

    if cfg.DATASETS.TYPE == 'coco_panoptic':
        warnings.warn("COCO panoptic has not been tested on this implementation", UserWarning)
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, cfg)

    if cfg.DATASETS.TYPE == 'vis':
        if cfg.DATASETS.DEVIS.COCO_JOINT_TRAINING and image_set == "TRAIN":
            vis_dataset, num_classes = build_vis(image_set, cfg)
            coco_vis_dataset = build_coco_joint_vis(cfg, num_classes)
            return ConcatDataset([coco_vis_dataset, vis_dataset]), num_classes

        return build_vis(image_set, cfg)

    raise ValueError(f'dataset type {cfg.DATASETS.TYPE} not supported')
