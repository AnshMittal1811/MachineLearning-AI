"""
Adapt from:
- https://github.com/svip-lab/PlanarReconstruction
- https://github.com/CSAILVision/semantic-segmentation-pytorch
"""

import os
import math
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


'''
MISC for 360 panorama
'''
class LR_PAD(nn.Module):
    def __init__(self, pad):
        super(LR_PAD, self).__init__()
        self.pad = pad

    def forward(self, x):
        return torch.cat([x[..., -self.pad:], x, x[..., :self.pad]], dim=3)

def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )


"""
Backbone
"""
model_urls = {
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}

def load_url(url):
    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    os.makedirs(model_dir, exist_ok=True)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.isfile(cached_file):
        urlretrieve(url, filename=cached_file)
    return torch.load(cached_file, map_location=lambda storage, loc: storage)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.bn3.weight.data.zero_()
        self.bn3.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5


def _resnet(layers, url, **kwargs):
    model = ResNet(Bottleneck, layers, **kwargs)
    model.load_state_dict(load_url(url), strict=True)
    del model.avgpool
    del model.fc
    return model


def resnet50(**kwargs):
    return _resnet([3, 4, 6, 3], model_urls['resnet50'], **kwargs)


def resnet101(**kwargs):
    return _resnet([3, 4, 23, 3], model_urls['resnet101'], **kwargs)


"""
Resnet-FPN
"""
class ResnetFPN(nn.Module):
    def __init__(self, backbone='resnet101', channel=256, lr_pad=True,
                 add_u=False, add_v=False, no_p0p1=False):
        super(ResnetFPN, self).__init__()

        if backbone == 'resnet101':
            self.backbone = resnet101()
        elif backbone == 'resnet50':
            self.backbone = resnet50()
        else:
            raise NotImplementedError()

        self.add_u = add_u
        self.add_v = add_v
        self.no_p0p1 = no_p0p1
        extra_in_ch = int(add_u) + int(add_v)
        if extra_in_ch > 0:
            ori_conv1 = self.backbone.conv1
            new_conv1 = nn.Conv2d(
                3+extra_in_ch, ori_conv1.out_channels,
            	kernel_size=ori_conv1.kernel_size,
            	stride=ori_conv1.stride,
            	padding=ori_conv1.padding,
            	bias=ori_conv1.bias)
            with torch.no_grad():
                new_conv1.weight[:, 3:].normal_(std=ori_conv1.weight.std().item()/5)
            self.backbone.conv1 = new_conv1

        self.relu = nn.ReLU(inplace=True)

        # lateral
        self.c5_conv = nn.Conv2d(2048, channel, (1, 1))
        self.c4_conv = nn.Conv2d(1024, channel, (1, 1))
        self.c3_conv = nn.Conv2d(512, channel, (1, 1))
        self.c2_conv = nn.Conv2d(256, channel, (1, 1))
        if self.no_p0p1:
            self.c1_conv = nn.Identity()
            self.p0_conv = nn.Identity()
        else:
            self.c1_conv = nn.Conv2d(128, channel, (1, 1))
            self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)

        self.lr_pad = lr_pad
        if lr_pad:
            wrap_lr_pad(self)

    def forward(self, x):
        B, C, H_, W = x.shape
        if self.add_u:
            u_coord = torch.linspace(-1,1,W).to(x.device).reshape(1,1,1,W).repeat(B,1,H_,1)
            x = torch.cat([x, u_coord], 1)
        if self.add_v:
            v_coord = torch.linspace(-1,1,H_).to(x.device).reshape(1,1,H_,1).repeat(B,1,1,W)
            x = torch.cat([x, v_coord], 1)

        # bottom up
        c1, c2, c3, c4, c5 = self.backbone(x)

        # top down
        p5 = self.c5_conv(c5)
        p4 = self.c4_conv(c4) + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=True)
        p3 = self.c3_conv(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=True)
        p2 = self.c2_conv(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True)

        if self.no_p0p1:
            p0, p1 = p2, p2
        else:
            p1 = self.c1_conv(c1) + F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=True)
            p0 = F.interpolate(p1, scale_factor=2, mode='bilinear', align_corners=True)
            p0 = self.p0_conv(p0)

        return p0, p1, p2, p3, p4, p5


if __name__ == '__main__':
    model = ResnetFPN(backbone='resnet101', channel=64)
