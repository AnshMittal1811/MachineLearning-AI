import numpy as np
import functools

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu
from torch.autograd import Variable

from torch.utils.serialization import load_lua

from .ClassicSRNet import SRResNet_Residual_Block
from .modules import DownsamplingShuffle, Features4Layer, Features3Layer, Attention

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)


def GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=-1.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(np.random.normal(loc=mean, scale=sigma, size=size))
    return torch.clamp(noise + tensor, min=min, max=max)


def PoissonNoising(tensor, lamb, noise_size=None, min=-1.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(np.random.poisson(lam=lamb, size=size))
    return torch.clamp(noise + tensor, min=min, max=max)


class DeNet1641(nn.Module):
    """
    1641 parameters
    No residual
    Leaky ReLU

    Sequential(
      (0): ReflectionPad2d((2, 2, 2, 2))
      (1): Conv2d (1, 4, kernel_size=(5, 5), stride=(1, 1))
      (2): LeakyReLU(0.15)
      (3): ReflectionPad2d((1, 1, 1, 1))
      (4): Conv2d (4, 8, kernel_size=(3, 3), stride=(1, 1))
      (5): LeakyReLU(0.15)
      (6): ReflectionPad2d((1, 1, 1, 1))
      (7): Conv2d (8, 8, kernel_size=(3, 3), stride=(1, 1))
      (8): LeakyReLU(0.15)
      (9): ReflectionPad2d((1, 1, 1, 1))
      (10): Conv2d (8, 8, kernel_size=(3, 3), stride=(1, 1))
      (11): LeakyReLU(0.15)
      (12): ReflectionPad2d((1, 1, 1, 1))
      (13): Conv2d (8, 1, kernel_size=(3, 3), stride=(1, 1))
    )
    """
    def __init__(self):
        super(DeNet1641, self).__init__()
        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU(0.15)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(0.15)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.LeakyReLU(0.15)
        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.LeakyReLU(0.15)
        self.pad5 = nn.ReflectionPad2d(1)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.tanh = F.tanh

    def forward(self, input):
        return self.tanh(self.conv5(self.pad5(
            self.relu4(self.conv4(self.pad4(
                self.relu3(self.conv3(self.pad3(
                    self.relu2(self.conv2(self.pad2(
                        self.relu1(self.conv1(self.pad1(input)))
                    )))
                )))
            )))
        )))


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ICCVGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', residual_learning=True, min=0.0, max=1.0):
        assert (n_blocks >= 0)
        super(ICCVGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.min = min
        self.max = max
        self.residual_learning = residual_learning
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0,
                           bias=use_bias),
                 nn.ReLU(True),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(ngf, ngf, kernel_size=3, padding=0,
                           bias=use_bias, stride=1),
                 norm_layer(ngf),
                 nn.ReLU(True)
                 ]
        self.model_fore = nn.Sequential(*model)

        model = []
        for i in range(n_blocks):
            model.append(ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                     use_bias=use_bias))
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, ngf, 3)]
        self.model_body = nn.Sequential(*model)

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(ngf, ngf, 3),
                 nn.ReLU(True),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(ngf, 1, 3)
                 ]

        self.model_tail = nn.Sequential(*model)

    def forward(self, input):
        input_ori = input.clone()
        input = self.model_fore(input)
        input_ = input.clone()
        input = self.model_body(input)
        input += input_
        input = self.model_tail(input)
        return torch.clamp(input, min=self.min, max=self.max) if not self.residual_learning else torch.clamp(input + input_ori, min=self.min, max=self.max)


class NoiseGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, sigma=0.03, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', residual_learning=True, min=0.0, max=1.0):
        assert (n_blocks >= 0)
        super(NoiseGenerator, self).__init__()
        self.input_nc = input_nc + 1
        self.output_nc = output_nc
        self.ngf = ngf
        self.min = min
        self.max = max
        self.sigma = sigma
        self.residual_learning = residual_learning
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc + 1, ngf, kernel_size=3, padding=0,
                           bias=use_bias),
                 nn.ReLU(True),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(ngf, ngf, kernel_size=3, padding=0,
                           bias=use_bias, stride=1),
                 norm_layer(ngf),
                 nn.ReLU(True)
                 ]
        self.model_fore = nn.Sequential(*model)

        model = []
        for i in range(n_blocks):
            model.append(ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                     use_bias=use_bias))
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, ngf, 3)]
        self.model_body = nn.Sequential(*model)

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(ngf, ngf, 3),
                 nn.ReLU(True),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(ngf, 1, 3)
                 ]

        self.model_tail = nn.Sequential(*model)

    def forward(self, input_c):
        B,C,H,W = input_c.size()
        input_n = Variable(GaussianNoising(torch.zeros(B, C, H, W), self.sigma, min=0.0, max=1.0))
        if isinstance(input_c.data, torch.cuda.FloatTensor):
            input_n = input_n.cuda()
        input_list = [input_c, input_n]
        input = torch.cat(input_list, dim=1)
        input_ori = input_c.clone()
        input = self.model_fore(input)
        input_ = input.clone()
        input = self.model_body(input)
        input += input_
        input = self.model_tail(input)
        return torch.clamp(input, min=self.min, max=self.max) if not self.residual_learning else torch.clamp(input + input_ori, min=self.min, max=self.max)


class Discriminator(nn.Module):
    def __init__(self, sigmoid=False):
        super(Discriminator, self).__init__()
        self.sigmoid = sigmoid
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        if self.sigmoid:
            return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
        else:
            return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)]  #TODO: if pixel can not load weights, check this padding, use to be 1, modified to 0

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class PixelDiscriminatorDualPath(nn.Module):
    def __init__(self, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(PixelDiscriminatorDualPath, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.localnet = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            ]

        self.globalnet = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            ]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.localnet = nn.Sequential(*self.localnet)

    def forward(self, input):
        return self.net(input)


class AttentionPixelDiscriminator(nn.Module):
    """
    Discriminator with attention module
    """
    def __init__(self, input_nc=1, ndf=64, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, feature_channels=16, down_samples=2):
        super(AttentionPixelDiscriminator, self).__init__()
        self.pixelD = PixelDiscriminator(input_nc=input_nc, ndf=ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        self.attention = Attention(input_channel=input_nc, feature_channels=feature_channels, down_samples=down_samples)

    def forward(self, input, train=False, is_attention=False):
        pixel_level_pred = self.pixelD(input)
        if train:
            attention_map = self.attention(input)
        else:
            attention_map = self.attention(input).detach()
        weighted_pred = torch.mul(pixel_level_pred, attention_map)
        weighted_sum = torch.sum(weighted_pred.view(weighted_pred.size(0), weighted_pred.size(1), -1), dim=2)
        attention_sum = torch.sum(attention_map.view(attention_map.size(0), attention_map.size(1), -1), dim=2)
        final_pred = torch.div(weighted_sum, attention_sum)
        return_tuple = final_pred
        if train:
            return final_pred, pixel_level_pred
        elif is_attention:
            return final_pred, pixel_level_pred, attention_map
        else:
            return return_tuple


class DSRCNN_Denoise_BN(nn.Module):
    def __init__(self):
        super(DSRCNN_Denoise_BN, self).__init__()
        self.conv1_ref = nn.Conv2d(1, 4, 5, padding=2)
        self.relu1_ref = nn.PReLU(num_parameters=4)
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.relu1 = nn.PReLU(num_parameters=4)
        self.conv2 = nn.Conv2d(16, 8, 1, padding=0)
        self.relu2 = nn.PReLU(num_parameters=8)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.PReLU(num_parameters=8)
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=8)
        self.conv5 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True)
        self.blob_conv1_ref = []
        self.blob_eletsum = []

    def forward(self, x):
        split_dim = 1
        split_x = torch.split(tensor=x, split_size=2, dim=split_dim)
        split_sum = torch.sum(split_x[0], dim=split_dim, keepdim=True)
        conv1_x = []
        self.blob_conv1_ref = self.relu1_ref(self.conv1_ref(split_sum))
        conv1_x.append(self.relu1_ref(self.conv1_ref(split_sum)))
        for i in range(1, len(split_x)):
            split_sum = torch.sum(split_x[i], dim=split_dim, keepdim=True)
            conv1_x.append(self.relu1(self.conv1(split_sum)))
        x = torch.cat(conv1_x, dim=split_dim)
        x = self.relu2(self.conv2(x))
        x_res = self.relu3(self.conv3_bn(self.conv3(x)))
        x_res = self.relu4(self.conv4_bn(self.conv4(x_res)))
        x_res = self.conv5_bn(self.conv5(x_res))
        x = x + x_res

        self.blob_eletsum = x

        return x


class DSRCNN_SR_x4(nn.Module):
    def __init__(self):
        super(DSRCNN_SR_x4, self).__init__()
        self.conv6 = nn.Conv2d(8, 4, 1, padding=0)
        self.relu6 = nn.PReLU(num_parameters=4)
        self.deconv7 = nn.ConvTranspose2d(4, 1, 8, stride=4, padding=2)
        self.blob_output = []

    def forward(self, x):
        x = self.relu6(self.conv6(x))
        x = self.deconv7(x)
        return torch.clamp(x, min=0.0, max=1.0)


class DSRCNN_Denoise_BN_Large(nn.Module):
    def __init__(self):
        super(DSRCNN_Denoise_BN_Large, self).__init__()
        self.conv1_ref = nn.Conv2d(1, 32, 5, padding=2)
        self.relu1_ref = nn.PReLU(num_parameters=32)
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.relu1 = nn.PReLU(num_parameters=32)
        self.conv2 = nn.Conv2d(128, 64, 1, padding=0)
        self.relu2 = nn.PReLU(num_parameters=64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.PReLU(num_parameters=64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.PReLU(num_parameters=64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.blob_conv1_ref = []
        self.blob_eletsum = []

    def forward(self, x):
        split_dim = 1
        split_x = torch.split(tensor=x, split_size=2, dim=split_dim)
        split_sum = torch.sum(split_x[0], dim=split_dim, keepdim=True)
        conv1_x = []
        self.blob_conv1_ref = self.relu1_ref(self.conv1_ref(split_sum))
        conv1_x.append(self.relu1_ref(self.conv1_ref(split_sum)))
        for i in range(1, len(split_x)):
            split_sum = torch.sum(split_x[i], dim=split_dim, keepdim=True)
            conv1_x.append(self.relu1(self.conv1(split_sum)))
        x = torch.cat(conv1_x, dim=split_dim)
        x = self.relu2(self.conv2(x))
        x_res = self.relu3(self.conv3_bn(self.conv3(x)))
        x_res = self.relu4(self.conv4_bn(self.conv4(x_res)))
        x_res = self.conv5_bn(self.conv5(x_res))
        x = x + x_res

        self.blob_eletsum = x

        return x


class DSRCNN_SR_x4_Large(nn.Module):
    def __init__(self):
        super(DSRCNN_SR_x4_Large, self).__init__()
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu6 = nn.PReLU(num_parameters=64)
        self.conv7 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        self.pix = nn.PixelShuffle(4)
        self.blob_output = []

    def forward(self, x):
        x = self.relu6(self.conv6(x))
        x = self.pix(self.conv7(x))
        return torch.clamp(x, min=0.0, max=1.0)


class DSRCNN_SR_x4_LargeDeconv(nn.Module):
    def __init__(self):
        super(DSRCNN_SR_x4_LargeDeconv, self).__init__()
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu6 = nn.PReLU(num_parameters=64)
        self.deconv7 = nn.ConvTranspose2d(64, 1, 8, stride=4, padding=2)
        self.blob_output = []

    def forward(self, x):
        x = self.relu6(self.conv6(x))
        x = self.deconv7(x)
        return torch.clamp(x, min=0.0, max=1.0)


class DSRCNN_SR_x4_DeepLargeRes(nn.Module):
    def __init__(self):
        super(DSRCNN_SR_x4_DeepLargeRes, self).__init__()
        self.feature_ref = Features4Layer(features=32)
        self.feature_mul = Features4Layer(features=32)
        self.shrink_conv = nn.Conv2d(32 * 8, 64, kernel_size=3, stride=1, padding=1)
        self.shrink_relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        split_x = torch.split(tensor=input, split_size=1, dim=1)
        conv1_x = []
        conv1_x.append(self.feature_ref(split_x[0]))
        for i in range(1, len(split_x)):
            conv1_x.append(self.feature_mul(split_x[i]))
        shrink = torch.cat(conv1_x, dim=1)
        shrinked = self.shrink_relu(self.shrink_conv(shrink))
        resed = self.residual(shrinked)
        up = self.bn_mid(self.conv_mid(resed))
        return self.conv_output(self.upscale4x(up))


class DSRCNN_SR_x4_DeepLargeRes_FeatureFusion(nn.Module):
    def __init__(self):
        super(DSRCNN_SR_x4_DeepLargeRes_FeatureFusion, self).__init__()
        # self.feature_ref = Features4Layer(features=32)
        self.feature_mul = Features4Layer(features=32)
        self.shrink_conv = nn.Conv2d(32 * 2, 64, kernel_size=3, stride=1, padding=1)
        self.shrink_relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        B, F, H, W = input.size()
        split_x = torch.split(tensor=input, split_size=1, dim=1)
        conv1_x = []
        conv1_x.append(self.feature_mul(split_x[0]))
        for i in range(1, len(split_x)):
            conv1_x.append(self.feature_mul(split_x[i]))
        fusion = sum(conv1_x) / F
        shrink = torch.cat([fusion, conv1_x[0]], dim=1)
        shrinked = self.shrink_relu(self.shrink_conv(shrink))
        resed = self.residual(shrinked)
        up = self.bn_mid(self.conv_mid(resed))
        return self.conv_output(self.upscale4x(up))


class DSRCNN_preFusion(nn.Module):
    def __init__(self):
        super(DSRCNN_preFusion, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
