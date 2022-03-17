# coding: utf-8
import numpy as np
import random
import functools

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io
import pdb

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

import math

from .modules import residualBlock, upsampleBlock, DownsamplingShuffle, Attention, Flatten

def PCA(data, k=15):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return torch.mm(X, U[:, :k])


class PCAEncoder(object):
    def __init__(self, weight, cuda=False):
        # self.weight = torch.load(weight)
        data = scipy.io.loadmat(weight)
        self.weight = torch.t(torch.FloatTensor(data['pca']))
        self.size = self.weight.size()
        if cuda:
            self.weight = Variable(self.weight).cuda()
        else:
            self.weight = Variable(self.weight)

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size()
        result = torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size))
        return torch.squeeze(result)

    def decode(self, code):
        pass


class MDPCAEncoder(object):
    def __init__(self, weight, cuda=False):
        data = scipy.io.loadmat(weight)
        self.weight = torch.FloatTensor(data['pca'])
        self.size = self.weight.size()
        if cuda:
            self.weight = Variable(self.weight).cuda()
        else:
            self.weight = Variable(self.weight)
        # self.use_cuda = cuda

    def __call__(self, batch_kernel):
        # gaussian_param = gaussian_param.cuda() if self.use_cuda else gaussian_param
        B, C, H, W = batch_kernel.size()
        result = torch.bmm(batch_kernel.view((B, C, H * W)), self.weight.expand((B, ) + self.size))
        # result = torch.cat((torch.squeeze(result.data, 1), gaussian_param.data), dim=1)
        return torch.squeeze(result)

    def decode(self, code):
        pass


class SRMD_Block(nn.Module):
    def __init__(self, ch_in, bn=True):
        super(SRMD_Block, self).__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(ch_in, 128, 3, 1, 1),  # ch_in, ch_out, kernel_size, stride, pad
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(ch_in, 128, 3, 1, 1),  # ch_in, ch_out, kernel_size, stride, pad
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        return self.block(x)


class SRMD_Net(nn.Module):
    def __init__(self, scala=2, input_para=15, min=0.0, max=1.0, bn=True):
        super(SRMD_Net, self).__init__()
        self.scala = scala
        self.min = min
        self.max = max
        self.input_para = input_para
        self.bn = bn
        self.net = self.make_net()

    def make_net(self):
        layers = [
            SRMD_Block(self.input_para + 1, self.bn),
            SRMD_Block(128, self.bn),
            SRMD_Block(128, self.bn),
            SRMD_Block(128, self.bn),
            SRMD_Block(128, self.bn),

            nn.Conv2d(in_channels=128, out_channels=self.scala**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=self.scala)
        ]
        return nn.Sequential(*layers)

    def forward(self, input, code, clip=False):
        B, C, H, W = input.size()
        # B, C_l, H_l = code.size()
        B, C_l= code.size()
        # code_exp = code.view((B, C_l * H_l, 1, 1)).expand((B, H_l, H, W))
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))
        cat_input = torch.cat([input, code_exp], dim=1)
        result = self.net(cat_input)
        return result
        # return torch.clamp(self.convf(x), min=self.min, max=self.max) if clip else self.convf(x)


class FSRCNNY_MD(nn.Module):
    def __init__(self, scala=2, input_para=15, min=0.0, max=1.0):
        super(FSRCNNY_MD, self).__init__()
        self.scala = scala
        self.min = min
        self.max = max

        self.conv1 = nn.Conv2d(in_channels=1 + input_para, out_channels=64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1),
                            upsampleBlock(64, 64 * 4, activation=nn.LeakyReLU(0.2, inplace=True)))

        self.convf = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, input, code, clip=False):
        B, C, H, W = input.size()
        B, C_l = code.size()
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))
        cat_input = torch.cat([input, code_exp], dim=1)
        out = self.relu1(self.conv1(cat_input))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        x = self.relu6(self.conv6(out))

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return torch.clamp(self.convf(x), min=self.min, max=self.max) if clip else self.convf(x)


'''
NOTE:
    输入图像，经过卷积+pooling，从图像中解析出blur kernel的code
    用pooling应该是模拟PCA的过程
    卷积提特征（得到类似kernel），pooling降维，linear取前15个特征，得到kernel的code
'''
class CodeInit(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, pooling='avg', kernel_size=5):
        super(CodeInit, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ])

        if pooling == 'avg':
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.globalPooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.Dense = nn.Sequential(*[
            nn.Linear(ndf, ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf * 2, code_len),
        ])

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        flat_dense = flat.view(flat.size()[:2])
        dense = self.Dense(flat_dense)
        return dense


class CodeAgent(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, pooling='avg'):
        super(CodeAgent, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ])

        if pooling == 'avg':
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.globalPooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.code_dense = nn.Sequential(*[
            nn.Linear(code_len, ndf),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Linear(ndf * 2, ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf * 2, code_len)
        ])

    def forward(self, input, code):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        flat_dense = flat.view(flat.size()[:2])
        code_dense = self.code_dense(code)
        global_dense = torch.cat([flat_dense, code_dense], dim=1)
        code_residual = self.global_dense(global_dense)
        return torch.add(code, code_residual)


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.sizes = sizes

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # h, w = max(self.sizes), max(self.sizes)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class CodeEstimate(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, pooling='avg', kernel_size=3):
        super(CodeEstimate, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ])

        self.psp = PSPModule(ndf, ndf)

        if pooling == 'avg':
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.globalPooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.Dense = nn.Sequential(*[
            nn.Linear(ndf, ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf * 2, code_len),
        ])

    def forward(self, input):
        feat = self.ConvNet(input)
        p = self.psp(feat)
        flat = self.globalPooling(p)
        flat_dense = flat.view(flat.size()[:2])
        dense = self.Dense(flat_dense)
        return dense


class MapEstimator(nn.Module):
    def __init__(self, code_len=15, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, pooling='avg', kernel_size=5):
        super(MapEstimator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            # add for map estimation
            nn.Conv2d(ndf, code_len, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias),
            norm_layer(code_len),
            nn.LeakyReLU(0.2, True)
        ])

    def forward(self, input):
        map = self.ConvNet(input)
        return map

def dimension_stretch(tensor, vector):
    B, C, H, W = tensor.size()
    B, C_l = vector.size()
    return vector.view((B, C_l, 1, 1)).expand((B, C_l, H, W))


