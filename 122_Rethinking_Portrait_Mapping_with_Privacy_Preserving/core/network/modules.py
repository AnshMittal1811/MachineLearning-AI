"""
Rethinking Portrait Matting with Privacy Preserving

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting.git
Paper link: https://arxiv.org/abs/2203.16828

"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TFI(nn.Module):
    expansion = 1
    def __init__(self, planes,stride=1):
        super(TFI, self).__init__()
        middle_planes = int(planes/2)
        self.transform = conv1x1(planes, middle_planes)
        self.conv1 = conv3x3(middle_planes*3, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, input_s_guidance, input_m_decoder, input_m_encoder):
        input_s_guidance_transform = self.transform(input_s_guidance)
        input_m_decoder_transform = self.transform(input_m_decoder)
        input_m_encoder_transform = self.transform(input_m_encoder)
        x = torch.cat((input_s_guidance_transform,input_m_decoder_transform,input_m_encoder_transform),1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class SBFI(nn.Module):
    def __init__(self, planes,planes2,stride=1):
        # planes2, min dim
        super(SBFI, self).__init__()
        self.stride = stride
        self.transform1 = conv1x1(planes, int(planes/2))
        self.transform2 = conv1x1(planes2, int(planes/2))
        self.maxpool = nn.MaxPool2d(2, stride=stride)
        self.conv1 = conv3x3(planes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_m_decoder,e0):
        input_m_decoder_transform = self.transform1(input_m_decoder)
        e0_maxpool = self.maxpool(e0)
        e0_transform = self.transform2(e0_maxpool)
        x = torch.cat((input_m_decoder_transform,e0_transform),1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out+input_m_decoder
        return out


class DBFI(nn.Module):
    def __init__(self, planes,planes2,stride=1):
        # planes2, max dim
        super(DBFI, self).__init__()
        self.stride = stride
        self.transform1 = conv1x1(planes, int(planes/2))
        self.transform2 = conv1x1(planes2, int(planes/2))
        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        self.conv1 = conv3x3(planes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, 3, 1)
        self.upsample2 = nn.Upsample(scale_factor=int(32/stride), mode='bilinear')

    def forward(self, input_s_decoder,e4):
        input_s_decoder_transform = self.transform1(input_s_decoder)
        e4_transform = self.transform2(e4)
        e4_upsample = self.upsample(e4_transform)
        x = torch.cat((input_s_decoder_transform,e4_upsample),1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out+input_s_decoder
        out_side = self.conv2(out)
        out_side = self.upsample2(out_side)
        return out, out_side
