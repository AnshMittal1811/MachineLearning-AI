# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch.nn as nn
from models.layers.normalization import LinearNoiseLayer


def spectral_conv_function(in_c, out_c, k, p, s):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s)
    )


def conv_function(in_c, out_c, k, p, s):
    return nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s)


def get_conv_layer(opt):
    if "spectral" in opt.norm_G:
        conv_layer_base = spectral_conv_function
    else:
        conv_layer_base = conv_function

    return conv_layer_base


# Convenience passthrough function
class Identity(nn.Module):
    def forward(self, input):
        return input


# ResNet Blocks
class ResNet_Block(nn.Module):
    def __init__(self, in_c, in_o, opt, downsample=None, norm="batch"):
        super().__init__()

        if "batch" in norm:
            norm_1 = LinearNoiseLayer(opt, output_sz=in_c)
            norm_2 = LinearNoiseLayer(opt, output_sz=in_o)
        elif "instance" in norm:
            norm_1 = nn.InstanceNorm2d(in_c)
            norm_2 = nn.InstanceNorm2d(in_o)
        conv_layer = get_conv_layer(opt)

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)
        conv_ab = conv_layer(in_o, in_o, 3, 1, 1)

        conv_b = conv_layer(in_c, in_o, 1, 0, 1)

        if downsample == "Down":
            norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = nn.Upsample(scale_factor=2, mode="bilinear")
        elif downsample:
            norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = Identity()

        self.ch_a = nn.Sequential(
            norm_1,
            nn.ReLU(),
            conv_aa,
            norm_2,
            nn.ReLU(),
            conv_ab,
            norm_downsample,
        )

        if downsample or (in_c != in_o):
            self.ch_b = nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = Identity()

    def forward(self, x):
        x_a = self.ch_a(x)
        x_b = self.ch_b(x)

        return x_a + x_b
