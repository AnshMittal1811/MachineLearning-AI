import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchmeta.modules import (MetaModule, MetaSequential)
import modules

from collections import OrderedDict
from torchmeta.modules.utils import get_subdict

"""
This module contains implementations of the UNet and Residual UNet architectures
used as the image encoder and decoder in MetaNLR++. These architectures and their 
implementations are heavily based upon those from Stable View Synthesis:
(https://github.com/isl-org/StableViewSynthesis)
"""


class ImNormalizer(object):
    def __init__(self, in_fmt="-11"):
        self.in_fmt = in_fmt
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def apply(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        if self.in_fmt == "-11":
            x = (x + 1) / 2
        elif self.in_fmt != "01":
            raise Exception("invalid input format")
        return (x - self.mean) / self.std


class ResUNet(nn.Module):
    def __init__(
        self, out_channels_0=64, out_channels=-1, depth=5, resnet="resnet18"
    ):
        """
        Residual U-Net architecture, adapted from Stable View Synthesis
        (https://github.com/isl-org/StableViewSynthesis)
        """
        super().__init__()

        if resnet == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception("invalid resnet model")

        self.normalizer = ImNormalizer()

        if depth < 1 or depth > 5:
            raise Exception("invalid depth of UNet")

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        decs = nn.ModuleList()
        enc_channels = 0
        if depth == 5:
            encs.append(resnet.layer4)
            enc_translates.append(self.convrelu(512, 512, 1))
            enc_channels = 512
        if depth >= 4:
            encs.append(resnet.layer3)
            enc_translates.append(self.convrelu(256, 256, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 256, 256))
            enc_channels = 256
        if depth >= 3:
            encs.append(resnet.layer2)
            enc_translates.append(self.convrelu(128, 128, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 128, 128))
            enc_channels = 128
        if depth >= 2:
            encs.append(nn.Sequential(resnet.maxpool, resnet.layer1))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        if depth >= 1:
            encs.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        enc_translates.append(
            nn.Sequential(self.convrelu(3, 64), self.convrelu(64, 64))
        )
        decs.append(self.convrelu(enc_channels + 64, out_channels_0))

        self.encs = nn.ModuleList(reversed(encs))
        self.enc_translates = nn.ModuleList(reversed(enc_translates))
        self.decs = nn.ModuleList(decs)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                out_channels_0, out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    # disable batchnorm learning in self.encs
    def train(self, mode=True):
        super().train(mode=mode)
        if not mode:
            return
        for mod in self.encs.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad_(False)

    def forward(self, x, params=None):
        x = self.normalizer.apply(x)

        outs = [self.enc_translates[0](x)]
        for enc, enc_translates in zip(self.encs, self.enc_translates[1:]):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)
        x = outs.pop()

        if self.out_conv:
            x = self.out_conv(x)
        return x


class ResUNet_Meta(MetaModule):
    def __init__(
        self, out_channels_0=64, out_channels=-1, depth=5, resnet="resnet18"
    ):
        """
        Residual U-Net architecture, adapted from Stable View Synthesis
        (https://github.com/isl-org/StableViewSynthesis) to be compatible with
        torchmeta and meta-learning
        """
        super().__init__()

        if resnet == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception("invalid resnet model")

        self.normalizer = ImNormalizer()

        if depth < 1 or depth > 5:
            raise Exception("invalid depth of UNet")

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        decs = nn.ModuleList()
        enc_channels = 0
        if depth == 5:
            encs.append(resnet.layer4)
            enc_translates.append(self.convrelu(512, 512, 1))
            enc_channels = 512
        if depth >= 4:
            encs.append(resnet.layer3)
            enc_translates.append(self.convrelu(256, 256, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 256, 256))
            enc_channels = 256
        if depth >= 3:
            encs.append(resnet.layer2)
            enc_translates.append(self.convrelu(128, 128, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 128, 128))
            enc_channels = 128
        if depth >= 2:
            encs.append(nn.Sequential(resnet.maxpool, resnet.layer1))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        if depth >= 1:
            encs.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        enc_translates.append(
            nn.Sequential(self.convrelu(3, 64), self.convrelu(64, 64))
        )
        decs.append(self.convrelu(enc_channels + 64, out_channels_0))

        encs = nn.ModuleList(reversed(encs))
        self.encs, self.encs_module_sizes = convert_to_meta(encs)

        decs = nn.ModuleList(decs)
        self.decs, self.decs_module_sizes = convert_to_meta(decs)

        enc_translates = nn.ModuleList(reversed(enc_translates))
        self.enc_translates, self.enc_translates_module_sizes = convert_to_meta(enc_translates)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample_nearest = nn.Upsample(
            scale_factor=2, mode="nearest"
        )

        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = modules.BatchConv2d(
                out_channels_0, out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            modules.BatchConv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = self.normalizer.apply(x)

        enc_trans_c = 0
        encs_c = 0
        decs_c = 0

        outs = [self.forward_module_list(x, self.enc_translates[enc_trans_c:enc_trans_c+self.enc_translates_module_sizes[0]],
                                         enc_trans_c, params=get_subdict(params, 'enc_translates'))]
        enc_trans_c += self.enc_translates_module_sizes[0]
        for i in range(len(self.encs_module_sizes)):
            x = self.forward_module_list(x, self.encs[encs_c:encs_c+self.encs_module_sizes[i]], encs_c, params=get_subdict(params, 'encs'))
            encs_c += self.encs_module_sizes[i]

            outs.append(self.forward_module_list(x, self.enc_translates[enc_trans_c:enc_trans_c+self.enc_translates_module_sizes[i+1]],
                                                 enc_trans_c, params=get_subdict(params, 'enc_translates')))
            enc_trans_c += self.enc_translates_module_sizes[i+1]

        for i in range(len(self.decs_module_sizes)):
            x0, x1 = outs.pop(), outs.pop()

            x = torch.cat((self.upsample(x0), x1), dim=1)

            x = self.forward_module_list(x, self.decs[decs_c:decs_c+self.decs_module_sizes[i]],
                                         decs_c, params=get_subdict(params, 'decs'))
            decs_c += self.decs_module_sizes[i]

            outs.append(x)

        x = outs.pop()

        if self.out_conv:
            x = self.out_conv(x, params=get_subdict('out_conv'))
        return x

    def forward_module_list(self, x, module_list, start_idx, params=None):
        for i, module in enumerate(module_list):
            if isinstance(module, modules.BatchConv2d):
                x = module(x, params=get_subdict(params, str(start_idx+i)))
            else:
                x = module(x)

        return x


def single_module(module):
    return isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) \
           or isinstance(module, nn.ReLU) or isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)


def unpack_module_list(module_list):
    module_list_unpacked = []
    module_sizes = []
    prev_length = 0
    for module in module_list:
        if single_module(module):
            module_list_unpacked.append(module)
        elif isinstance(module, torchvision.models.resnet.BasicBlock):
            module_list_unpacked.extend([module.conv1, module.bn1, module.relu,
                                         module.conv2, module.bn2])
        else:
            module_list_unpacked.extend(unpack_module_list(module)[0])

        module_sizes.append(len(module_list_unpacked) - prev_length)
        prev_length = len(module_list_unpacked)
    return module_list_unpacked, module_sizes


def convert_to_partial(module_list):
    module_list_unpacked, module_sizes = unpack_module_list(module_list)
    module_list_new = []

    for module in module_list_unpacked:
        if isinstance(module, nn.Conv2d):
            conv_replace = PartialConv2d(module.in_channels, module.out_channels,
                                         module.kernel_size, module.stride, module.padding,
                                         bias=False, return_mask=True, multi_channel=True)
            conv_replace.weight = module.weight
            module_list_new.append(conv_replace)
        else:
            module_list_new.append(module)

    return nn.ModuleList(module_list_new), module_sizes


def convert_to_meta(module_list):
    module_list_unpacked, module_sizes = unpack_module_list(module_list)
    module_list_new = []

    for module in module_list_unpacked:
        if isinstance(module, nn.Conv2d):
            conv_replace = modules.BatchConv2d(module.in_channels, module.out_channels,
                                               module.kernel_size, module.stride, module.padding,
                                               bias=module.bias is not None)
            conv_replace.weight = module.weight
            if module.bias is not None:
                conv_replace.bias = module.bias
            module_list_new.append(conv_replace)
        else:
            module_list_new.append(module)

    return nn.ModuleList(module_list_new), module_sizes


class ResUNet_PartialConv(nn.Module):
    def __init__(
        self, out_channels_0=64, out_channels=-1, depth=5, resnet="resnet18"
    ):
        """
        Residual U-Net architecture, adapted from Stable View Synthesis
        (https://github.com/isl-org/StableViewSynthesis) but uses partial convolutions
        instead of standard convolutions
        """
        super().__init__()

        if resnet == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception("invalid resnet model")

        self.normalizer = ImNormalizer()

        if depth < 1 or depth > 5:
            raise Exception("invalid depth of UNet")

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        decs = nn.ModuleList()
        enc_channels = 0
        if depth == 5:
            encs.append(resnet.layer4)
            enc_translates.append(self.convrelu(512, 512, 1))
            enc_channels = 512
        if depth >= 4:
            encs.append(resnet.layer3)
            enc_translates.append(self.convrelu(256, 256, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 256, 256))
            enc_channels = 256
        if depth >= 3:
            encs.append(resnet.layer2)
            enc_translates.append(self.convrelu(128, 128, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 128, 128))
            enc_channels = 128
        if depth >= 2:
            encs.append(nn.Sequential(resnet.maxpool, resnet.layer1))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        if depth >= 1:
            encs.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        enc_translates.append(
            nn.Sequential(self.convrelu(3, 64), self.convrelu(64, 64))
        )
        decs.append(self.convrelu(enc_channels + 64, out_channels_0))

        encs = nn.ModuleList(reversed(encs))
        self.encs, self.encs_module_sizes = convert_to_partial(encs)

        decs = nn.ModuleList(decs)
        self.decs, self.decs_module_sizes = convert_to_partial(decs)

        enc_translates = nn.ModuleList(reversed(enc_translates))
        self.enc_translates, self.enc_translates_module_sizes = convert_to_partial(enc_translates)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample_nearest = nn.Upsample(
            scale_factor=2, mode="nearest"
        )

        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = PartialConv2d(
                out_channels_0, out_channels, kernel_size=1, padding=0,
                return_mask=True, multi_channel=True
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            PartialConv2d(in_channels, out_channels, kernel_size, padding=padding,
                          return_mask=True, multi_channel=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, mask_in):
        x = self.normalizer.apply(x)

        enc_trans_c = 0
        encs_c = 0
        decs_c = 0

        outs = [self.forward_module_list(x, mask_in, self.enc_translates[enc_trans_c:enc_trans_c+self.enc_translates_module_sizes[0]])]
        enc_trans_c += self.enc_translates_module_sizes[0]
        for i in range(len(self.encs_module_sizes)):
            x, mask_in = self.forward_module_list(x, mask_in, self.encs[encs_c:encs_c+self.encs_module_sizes[i]])
            encs_c += self.encs_module_sizes[i]

            outs.append(self.forward_module_list(x, mask_in, self.enc_translates[enc_trans_c:enc_trans_c+self.enc_translates_module_sizes[i+1]]))
            enc_trans_c += self.enc_translates_module_sizes[i+1]

        for i in range(len(self.decs_module_sizes)):
            (x0, mask_in0), (x1, mask_in1) = outs.pop(), outs.pop()

            x = torch.cat((self.upsample(x0), x1), dim=1)
            mask_in = torch.cat((self.upsample_nearest(mask_in0), mask_in1), dim=1)

            x, mask_in = self.forward_module_list(x, mask_in, self.decs[decs_c:decs_c+self.decs_module_sizes[i]])
            decs_c += self.decs_module_sizes[i]

            outs.append((x, mask_in))

        x, mask_in = outs.pop()

        if self.out_conv:
            x, mask_in = self.out_conv(x, mask_in)
        return x

    def forward_module_list(self, x, mask_in, module_list):
        for module in module_list:
            if isinstance(module, PartialConv2d):
                x, mask_in = module(x, mask_in)
            elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
                x = module(x)
                mask_in = module(mask_in)
            else:
                x = module(x)

        return x, mask_in


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        Partial convolution layer, adapted from
        (https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py)
        """
        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        enc_channels=[64, 128, 256],
        dec_channels=[128, 64],
        out_channels=3,
        n_enc_convs=2,
        n_dec_convs=2,
    ):
        """
        U-Net architecture, adapted from Stable View Synthesis
        (https://github.com/isl-org/StableViewSynthesis)
        """
        super().__init__()

        self.encs = nn.ModuleList()
        self.enc_translates = nn.ModuleList()
        pool = False
        for enc_channel in enc_channels:
            stage = self.create_stage(
                in_channels, enc_channel, n_enc_convs, pool
            )
            self.encs.append(stage)
            translate = nn.Conv2d(enc_channel, enc_channel, kernel_size=1)
            self.enc_translates.append(translate)
            in_channels, pool = enc_channel, True

        self.decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(
                in_channels, dec_channel, n_dec_convs, False
            )
            self.decs.append(stage)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs, pool):
        mods = []
        if pool:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, x, params=None):
        outs = []
        for enc, enc_translates in zip(self.encs, self.enc_translates):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)

        x = outs.pop()
        if self.out_conv:
            x = self.out_conv(x)
        return x


class UNet_PartialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        enc_channels=[64, 128, 256],
        dec_channels=[128, 64],
        out_channels=3,
        n_enc_convs=2,
        n_dec_convs=2,
    ):
        """
        U-Net architecture, adapted from Stable View Synthesis
        (https://github.com/isl-org/StableViewSynthesis) but using partial
        convolution layers
        """
        super().__init__()

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        pool = False
        for enc_channel in enc_channels:
            stage = self.create_stage(
                in_channels, enc_channel, n_enc_convs, pool
            )
            encs.append(stage)
            translate = PartialConv2d(enc_channel, enc_channel, kernel_size=1,
                                      return_mask=True, multi_channel=True)
            enc_translates.append(translate)
            in_channels, pool = enc_channel, True

        decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(
                in_channels, dec_channel, n_dec_convs, False
            )
            decs.append(stage)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample_nearest = nn.Upsample(
            scale_factor=2, mode="nearest"
        )

        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = PartialConv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0,
                return_mask=True, multi_channel=True
            )

        self.encs, self.encs_module_sizes = convert_to_partial(encs)
        self.decs, self.decs_module_sizes = convert_to_partial(decs)
        self.enc_translates, self.enc_translates_module_sizes = convert_to_partial(enc_translates)

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            PartialConv2d(in_channels, out_channels, kernel_size, padding=padding,
                          return_mask=True, multi_channel=True),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs, pool):
        mods = []
        if pool:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, x, mask_in):
        enc_trans_c = 0
        encs_c = 0
        decs_c = 0

        outs = []
        for i in range(len(self.encs_module_sizes)):
            x, mask_in = self.forward_module_list(x, mask_in, self.encs[encs_c:encs_c+self.encs_module_sizes[i]])
            encs_c += self.encs_module_sizes[i]

            outs.append(self.forward_module_list(x, mask_in, self.enc_translates[enc_trans_c:enc_trans_c+self.enc_translates_module_sizes[i]]))
            enc_trans_c += self.enc_translates_module_sizes[i]

        for i in range(len(self.decs_module_sizes)):
            (x0, mask_in0), (x1, mask_in1) = outs.pop(), outs.pop()

            x = torch.cat((self.upsample(x0), x1), dim=1)
            mask_in = torch.cat((self.upsample_nearest(mask_in0), mask_in1), dim=1)

            x, mask_in = self.forward_module_list(x, mask_in, self.decs[decs_c:decs_c+self.decs_module_sizes[i]])
            decs_c += self.decs_module_sizes[i]

            outs.append((x, mask_in))

        x, mask_in = outs.pop()

        if self.out_conv:
            x, mask_in = self.out_conv(x, mask_in)
        return x

    def forward_module_list(self, x, mask_in, module_list):
        for module in module_list:
            if isinstance(module, PartialConv2d):
                x, mask_in = module(x, mask_in)
            elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
                x = module(x)
                mask_in = module(mask_in)
            else:
                x = module(x)

        return x, mask_in


class UNet_Meta(MetaModule):
    def __init__(
        self,
        in_channels,
        enc_channels=[64, 128, 256],
        dec_channels=[128, 64],
        out_channels=3,
        n_enc_convs=2,
        n_dec_convs=2,
        opt=None,
    ):
        """
        U-Net architecture, adapted from Stable View Synthesis
        (https://github.com/isl-org/StableViewSynthesis) but compatible with
        torchmeta for meta-learning the initialization
        """
        super().__init__()
        # ablation
        # if opt.dataset_name == 'nlr':
        #     enc_channels=[32, 64]
        #     dec_channels=[32]
        #     out_channels=3
        #     n_enc_convs=1
        #     n_dec_convs=1

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        pool = False
        for enc_channel in enc_channels:
            stage = self.create_stage(
                in_channels, enc_channel, n_enc_convs, pool
            )
            encs.append(stage)
            translate = modules.BatchConv2d(enc_channel, enc_channel, kernel_size=1)
            enc_translates.append(translate)
            in_channels, pool = enc_channel, True

        decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(
                in_channels, dec_channel, n_dec_convs, False
            )
            decs.append(stage)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = modules.BatchConv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0
            )

        self.encs, self.encs_module_sizes = convert_to_meta(encs)
        self.decs, self.decs_module_sizes = convert_to_meta(decs)
        self.enc_translates, self.enc_translates_module_sizes = convert_to_meta(enc_translates)

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            modules.BatchConv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs, pool):
        mods = []
        if pool:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        enc_trans_c = 0
        encs_c = 0
        decs_c = 0

        outs = []
        for i in range(len(self.encs_module_sizes)):
            x = self.forward_module_list(x, self.encs[encs_c:encs_c+self.encs_module_sizes[i]], encs_c, params=get_subdict(params, 'encs'))
            encs_c += self.encs_module_sizes[i]

            outs.append(self.forward_module_list(x, self.enc_translates[enc_trans_c:enc_trans_c+self.enc_translates_module_sizes[i]],
                                                 enc_trans_c, params=get_subdict(params, 'enc_translates')))
            enc_trans_c += self.enc_translates_module_sizes[i]

        for i in range(len(self.decs_module_sizes)):
            x0, x1 = outs.pop(), outs.pop()

            x = torch.cat((self.upsample(x0), x1), dim=1)

            x = self.forward_module_list(x, self.decs[decs_c:decs_c+self.decs_module_sizes[i]], decs_c, params=get_subdict(params, 'decs'))
            decs_c += self.decs_module_sizes[i]

            outs.append(x)

        x = outs.pop()

        if self.out_conv:
            x = self.out_conv(x, params=get_subdict(params, 'out_conv'))
        return x

    def forward_module_list(self, x, module_list, start_idx, params=None):
        for i, module in enumerate(module_list):
            if isinstance(module, modules.BatchConv2d):
                x = module(x, params=get_subdict(params, str(start_idx+i)))
            else:
                x = module(x)

        return x


class pretrain_enc_dec(nn.Module):
    def __init__(self, opt):
        """
        Module to pre-train the encoder and decoder on some image dataset
        """
        super().__init__()

        if opt.convolution_type == 'partial':
            self.enc_net = ResUNet_PartialConv(out_channels_0=opt.model_image_encoder_features,
                                               depth=opt.model_image_encoder_depth)
            self.dec_net = UNet_PartialConv(in_channels=opt.model_image_encoder_features)
            self.partial = True
        elif opt.convolution_type == 'meta':
            self.enc_net = ResUNet_Meta(out_channels_0=opt.model_image_encoder_features,
                                        depth=opt.model_image_encoder_depth)
            self.dec_net = UNet_Meta(in_channels=opt.model_image_encoder_features)
            self.partial = False
        else:
            self.enc_net = ResUNet(out_channels_0=opt.model_image_encoder_features,
                                   depth=opt.model_image_encoder_depth)
            self.dec_net = UNet(in_channels=opt.model_image_encoder_features)
            self.partial = False

        if opt.dataset_type == "DatasetFlyingChairs2":
            self.warping_fn = self.OpticFlowWarp
        else:
            raise RuntimeError(f"Warping type for {opt.dataset_type} not yet implemented")

    def OpticFlowWarp(self, encoded_features, optical_flow):
        B, C, H, W = encoded_features.size()

        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W, 1).repeat(B, 1, 1, 1)
        yy = yy.view(1, H, W, 1).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 3).float().to(encoded_features.device)

        vgrid = grid + optical_flow.permute(0, 2, 3, 1)

        # scale grid to [-1,1]
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1].clone() / max(H - 1, 1) - 1.0

        output = F.grid_sample(encoded_features, vgrid, align_corners=True)
        mask = torch.ones_like(encoded_features)
        mask = F.grid_sample(mask, vgrid, align_corners=True)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, model_in):
        image_warped = self.warping_fn(model_in['img0'], model_in['flow10']) * (1 - model_in['occ10'])

        if self.partial:
            encoded_features = self.enc_net(model_in['img0'], mask_in=None)
            decoded_image = self.dec_net(encoded_features, mask_in=None)
        else:
            encoded_features = self.enc_net(model_in['img0'])
            decoded_image = self.dec_net(encoded_features)

        encoded_features_warped = self.warping_fn(encoded_features, model_in['flow10']) * (1 - model_in['occ10'])

        if self.partial:
            decoded_image_warped = self.dec_net(encoded_features_warped,
                                                mask_in=(1 - model_in['occ10']).expand_as(encoded_features_warped))
        else:
            decoded_image_warped = self.dec_net(encoded_features_warped)

        return {'image_warped': image_warped,
                'encoded_features': encoded_features,
                'decoded_image': decoded_image,
                'encoded_features_warped': encoded_features_warped,
                'decoded_image_warped': decoded_image_warped}

