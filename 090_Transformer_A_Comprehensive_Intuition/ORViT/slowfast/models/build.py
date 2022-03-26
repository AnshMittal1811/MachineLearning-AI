#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    if getattr(cfg.MODEL, "LOAD_IN_PRETRAIN", "") != "":
        from .utils import load_pretrained
        _cfg = {
            'url':cfg.MODEL.LOAD_IN_PRETRAIN,
            'first_conv': 'patch_embed.proj',
            'classifier': 'head.projection',
            'num_classes': cfg.MODEL.NUM_CLASSES,
        }
        load_pretrained(model,
                        num_classes=cfg.MODEL.NUM_CLASSES,
                        cfg=_cfg,
                        in_chans=3,
                        img_size=cfg.DATA.TRAIN_CROP_SIZE,
                        num_patches=(cfg.DATA.TRAIN_CROP_SIZE // cfg.MVIT.PATCH_STRIDE[-1]) * (cfg.DATA.TRAIN_CROP_SIZE // cfg.MVIT.PATCH_STRIDE[-2]),
                        # pretrained_model=cfg.MODEL.LOAD_IN_PRETRAIN,
                        )

    if cfg.ORVIT.ENABLE and cfg.ORVIT.ZERO_INIT_ORVIT:
        from slowfast.utils import misc
        misc.module_0_init(model.orvit_blocks)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
        )
    return model
