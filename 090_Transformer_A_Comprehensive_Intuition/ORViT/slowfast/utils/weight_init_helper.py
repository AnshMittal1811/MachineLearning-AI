#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Utility function for weight initialization"""

import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill

def get_layer(model, layer_name):
    """
    Return the targeted layer (nn.Module Object) given a hierarchical layer name,
    separated by ".".
    Args:
        model (model): model to get layers from.
        layer_name (str): name of the layer.
    Returns:
        prev_module (nn.Module): the layer from the model with `layer_name` name.
    """
    layer_ls = layer_name.split(".")
    prev_module = model
    for layer in layer_ls:
        prev_module = prev_module._modules[layer]

    return prev_module

def _init_vit_weights(m, skip = []):
    for n, w in m.named_parameters():
        layer = get_layer(m, ".".join(n.split(".")[:-1]))
        if any([m in n for m in skip]):
            continue
        elif isinstance(layer, nn.Linear):
            if n.endswith('weight'): nn.init.trunc_normal_(w, std=0.02)
            elif n.endswith('bias'): nn.init.constant_(w, 0)
        elif isinstance(layer, nn.LayerNorm):
            if n.endswith('weight'): nn.init.constant_(w, 1.0)
            elif n.endswith('bias'): nn.init.constant_(w, 0)
        else:
            nn.init.trunc_normal_(w, std=0.02)
def init_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
            if (
                hasattr(m, "transform_final_bn")
                and m.transform_final_bn
                and zero_init_final_bn
            ):
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()
