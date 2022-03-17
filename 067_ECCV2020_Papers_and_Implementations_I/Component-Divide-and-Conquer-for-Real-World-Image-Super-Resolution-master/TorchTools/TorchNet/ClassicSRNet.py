import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

import math

from .modules import residualBlock, upsampleBlock, DownsamplingShuffle


class SRResNet_Residual_Block(nn.Module):
    def __init__(self, norm_type='IN', channel=64, in_place=True):
        super(SRResNet_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=in_place)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False)
        if norm_type == 'IN':
            self.in1 = nn.InstanceNorm2d(channel, affine=True)
            self.in2 = nn.InstanceNorm2d(channel, affine=True)
        elif norm_type == 'BN':
            self.in1 = nn.BatchNorm2d(channel)
            self.in2 = nn.BatchNorm2d(channel)
        else:
            def identity(tensor):
                return tensor
            self.in1 = identity
            self.in2 = identity

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class FSRCNNY(nn.Module):
    """
    Sequential(
      (0): Conv2d (1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (1): LeakyReLU(0.2, inplace)
      (2): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (3): LeakyReLU(0.2, inplace)
      (4): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (5): LeakyReLU(0.2, inplace)
      (6): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (7): LeakyReLU(0.2, inplace)
      (8): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (9): LeakyReLU(0.2, inplace)
      (10): Conv2d (1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (11): LeakyReLU(0.2, inplace)
      (12): upsampleBlock(
        (act): LeakyReLU(0.2, inplace)
        (pad): ReflectionPad2d((1, 1, 1, 1))
        (conv): Conv2d (64, 256, kernel_size=(3, 3), stride=(1, 1))
        (shuffler): PixelShuffle(upscale_factor=2)
      )
      (13): upsampleBlock(
        (act): LeakyReLU(0.2, inplace)
        (pad): ReflectionPad2d((1, 1, 1, 1))
        (conv): Conv2d (64, 256, kernel_size=(3, 3), stride=(1, 1))
        (shuffler): PixelShuffle(upscale_factor=2)
      )
      (14): Conv2d (64, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    )
    """
    def __init__(self, scala=4):
        super(FSRCNNY, self).__init__()
        self.scala = scala
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
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
            self.add_module('upsample' + str(i + 1), upsampleBlock(64, 64 * 4, activation=nn.LeakyReLU(0.2, inplace=True)))

        self.convf = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=2, bias=False)

    def forward(self, input):
        out = self.relu1(self.conv1(input))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        x = self.relu6(self.conv6(out))

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return F.tanh(self.convf(x))


class SRResNet(nn.Module):
    """
    SRResNet, norm_type = ['IN' | 'BN' | None]
    """
    def __init__(self, scala=2, inc=3, tanh=False, mid_fea=False, norm_type=None):
        super(SRResNet, self).__init__()
        self.scala = scala
        self.tanh = tanh
        self.return_mid_fea = mid_fea
        self.norm_type = norm_type

        self.conv_input = nn.Conv2d(in_channels=inc, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1),
                            upsampleBlock(64, 64 * 4, activation=nn.LeakyReLU(0.2, inplace=True)))

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=inc, kernel_size=9, stride=1, padding=4, bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.norm_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)

        lr_feature = out
        for i in range(int(log2(self.scala))):
            out = self.__getattr__('upsample' + str(i + 1))(out)
        hr_feature = out
        out = self.conv_output(out)

        if self.return_mid_fea:
            return out, lr_feature, hr_feature
        else:
            return out


class SRResNet_C2F(nn.Module):
    """
    SRResNet, norm_type = ['IN' | 'BN' | None]
    """
    def __init__(self, scala=2, inc=3, mid_fea=False, norm_type=None):
        super(SRResNet_C2F, self).__init__()
        self.scala = scala
        self.return_mid_fea = mid_fea
        self.norm_type = norm_type
        self.in_place = False

        self.conv_input = nn.Conv2d(in_channels=inc, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=self.in_place)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1),
                            upsampleBlock(64, 64 * 4, activation=nn.LeakyReLU(0.2, inplace=self.in_place)))

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=inc, kernel_size=9, stride=1, padding=4, bias=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(norm_type=self.norm_type, in_place=self.in_place))
        return nn.Sequential(*layers)

    def forward(self, x, global_fea=None, global_lambda=1., local_lambda=1.):
        out = self.relu(self.conv_input(x))

        if global_fea is not None:
            out = out * local_lambda + global_fea * global_lambda

        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)

        lr_feature = out
        for i in range(int(log2(self.scala))):
            out = self.__getattr__('upsample' + str(i + 1))(out)
        hr_feature = out
        out = self.conv_output(out)

        if self.return_mid_fea:
            return out, lr_feature, hr_feature
        else:
            return out


class SRResNet_LocalEnhancer(nn.Module):
    """
    SRResNet, norm_type = ['IN' | 'BN' | None]
    """
    def __init__(self, scala=2, inc=3, mid_fea=False, norm_type=None):
        super(SRResNet_LocalEnhancer, self).__init__()
        self.scala = scala
        self.return_mid_fea = mid_fea
        self.norm_type = norm_type
        self.in_place = False

        self.conv_input = nn.Conv2d(in_channels=inc, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=self.in_place)

        # Fuse Global Feature with Local
        fusion = [nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False),
                  nn.LeakyReLU(0.2, inplace=self.in_place)]
        self.fusion = nn.Sequential(*fusion)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1),
                            upsampleBlock(64, 64 * 4, activation=nn.LeakyReLU(0.2, inplace=self.in_place)))

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=inc, kernel_size=9, stride=1, padding=4, bias=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(norm_type=self.norm_type, in_place=self.in_place))
        return nn.Sequential(*layers)

    def forward(self, x, global_fea=None, global_lambda=1., local_lambda=1.):
        out = self.relu(self.conv_input(x))

        out = torch.cat([out, global_fea], dim=1)
        out = self.fusion(out)

        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)

        lr_feature = out
        for i in range(int(log2(self.scala))):
            out = self.__getattr__('upsample' + str(i + 1))(out)
        hr_feature = out
        out = self.conv_output(out)

        if self.return_mid_fea:
            return out, lr_feature, hr_feature
        else:
            return out


class Downsampler(nn.Sequential):
    """
    Downsampler for degrade model, use conv stride to downsample
    """
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1, bias=bias))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=5, stride=3, padding=1, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*m)


class Downsampler_conv(nn.Sequential):
    """
    Downsampler for degrade model, use conv stride to downsample
    """
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
                m.append(nn.ReflectionPad2d(1))
                m.append(nn.Conv2d(n_feats, n_feats, kernel_size=4, stride=2, padding=0, bias=bias))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(False))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=5, stride=3, padding=1, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Downsampler_conv, self).__init__(*m)


class DownSampleResNet(nn.Module):
    """
    Degradation model, can downsample input to 1/scale
    """
    def __init__(self, scala=4, inc=3, n_res_block=8, mid_fea=False):
        super(DownSampleResNet, self).__init__()
        self.scala = scala
        self.return_mid_fea = mid_fea

        self.conv_input = nn.Conv2d(in_channels=inc, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, n_res_block)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        m_tail = [
            # Downsampler(scala, 64, act=True),
            Downsampler_conv(scala, 64, act=True),
        ]

        self.conv_down = nn.Sequential(*m_tail)
        self.conv_output = nn.Conv2d(64, inc, kernel_size=3, padding=1)

        # self.conv_output = nn.Sequential(*m_tail)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)

        hr_feature = out
        out = self.conv_down(out)
        lr_feature = out
        out = self.conv_output(out)
        if self.return_mid_fea:
            return out, lr_feature, hr_feature
        else:
            return out
        # return torch.clamp(out, min=0., max=1.) if clip else out


class SRResNetRGBX4(nn.Module):
    def __init__(self, min=0.0, max=1.0, tanh=False):
        super(SRResNetRGBX4, self).__init__()
        self.min = min
        self.max = max
        self.tanh = tanh

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
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

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, clip=True):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        if self.tanh:
            return F.tanh(out)
        else:
            return torch.clamp(out, min=self.min, max=self.max) if clip else out


class SRResNetY(nn.Module):
    def __init__(self, input_para=0, scala=4, tanh=True, norm_type=None):
        super(SRResNetY, self).__init__()
        self.tanh = tanh

        self.conv_inputY = nn.Conv2d(in_channels=1 + input_para, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        if scala == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scala == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64 * 9, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_outputY = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, code=None, clip=True):
        if code is not None:
            B, C, H, W = x.size()
            B, C_l = code.size()
            code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))
            x = torch.cat([x, code_exp], dim=1)
        out = self.relu(self.conv_inputY(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale(out)
        out = self.conv_outputY(out)
        return out


class SRResNetYX2(nn.Module):
    def __init__(self, min=0.0, max=1.0, tanh=True):
        super(SRResNetYX2, self).__init__()
        self.min = min
        self.max = max
        self.tanh = tanh

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale2x = nn.Sequential(
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

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        if self.tanh:
            return F.tanh(out)
        else:
            return torch.clamp(out, min=self.min, max=self.max)


class DownSampleResNetYX4(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(DownSampleResNetYX4, self).__init__()
        self.min = min
        self.max = max
        self.down_shuffle = DownsamplingShuffle(4)

        self.conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 6)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(self.down_shuffle(x)))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.conv_output(out)
        return torch.clamp(out, min=self.min, max=self.max)


class SRResEnhancer(nn.Module):
    """
    Degrader or Enhancer, Won`t Change Image Resolution
    """
    def __init__(self, scala=2, inc=3, tanh=False, mid_fea=False, norm_type=None):
        super(SRResEnhancer, self).__init__()
        self.scala = scala
        self.tanh = tanh
        self.return_mid_fea = mid_fea
        self.norm_type = norm_type

        self.conv_input = nn.Conv2d(in_channels=inc, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(SRResNet_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=inc, kernel_size=9, stride=1, padding=4, bias=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.norm_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.conv_output(out)
        return out



# ===========================Previous Code=================================

# class SRResNet_Residual_Block_modi(nn.Module):
#     def __init__(self, norm_type=None):
#         super(SRResNet_Residual_Block_modi, self).__init__()
#
#         conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         relu1 = nn.LeakyReLU(0.2, inplace=True)
#         conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#
#         if norm_type == None:
#             self.res = [conv1, relu1, conv2]
#         elif norm_type == 'IN':
#             norm1 = nn.InstanceNorm2d(64, affine=True)
#             norm2 = nn.InstanceNorm2d(64, affine=True)
#             self.res = [conv1, norm1, relu1, conv2, norm2]
#         elif norm_type == 'BN':
#             norm1 = nn.BatchNorm2d(64)
#             norm2 = nn.BatchNorm2d(64)
#             self.res = [conv1, norm1, relu1, conv2, norm2]
#         else:
#             self.res = [conv1, relu1, conv2]
#         self.res = nn.Sequential(*self.res)
#
#     def forward(self, x):
#         res = self.res(x)
#         return torch.add(res, x)

# class SRResNet_Residual_Block(nn.Module):
#     def __init__(self, norm_type=None):
#         super(SRResNet_Residual_Block, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.in1 = nn.InstanceNorm2d(64, affine=True)
#         self.relu = nn.LeakyReLU(0.2, inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.in2 = nn.InstanceNorm2d(64, affine=True)
#
#     def forward(self, x):
#         identity_data = x
#         output = self.relu(self.in1(self.conv1(x)))
#         output = self.in2(self.conv2(output))
#         output = torch.add(output, identity_data)
#         return output
