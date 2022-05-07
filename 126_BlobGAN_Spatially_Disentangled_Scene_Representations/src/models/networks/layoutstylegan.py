# https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from utils import splat_features_from_scores
from .op import FusedLeakyReLU, conv2d_gradfix

__all__ = ["LayoutStyleGANGenerator"]

from .stylegan import PixelNorm, EqualLinear, ConstantInput, Blur, NoiseInjection, Upsample


@dataclass(eq=False)
class LayoutStyleGANGenerator(nn.Module):
    size: int = 256
    style_dim: int = 512
    n_mlp: int = 8
    channel_multiplier: int = 2
    lr_mlp: float = 0.01
    c_out: int = 3
    c_model: int = 64
    size_in: int = 16
    const_in: bool = False
    spatial_style: bool = False
    override_c_in: Optional[int] = None

    def __post_init__(self):
        super().__init__()

        blur_kernel = [1, 3, 3, 1]

        if not self.spatial_style:
            layers = [PixelNorm()]

            for i in range(self.n_mlp):
                layers.append(
                    EqualLinear(
                        self.style_dim, self.style_dim, lr_mul=self.lr_mlp, activation="fused_lrelu"
                    )
                )

            self.style = nn.Sequential(*layers)

        self.channels = {
            4: self.c_model * 8,
            8: self.c_model * 8,
            16: self.c_model * 8,
            32: self.c_model * 8,
            64: self.c_model * 4 * self.channel_multiplier,
            128: self.c_model * 2 * self.channel_multiplier,
            256: self.c_model * self.channel_multiplier,
            512: self.c_model // 2 * self.channel_multiplier,
            1024: self.c_model // 4 * self.channel_multiplier,
        }

        if self.const_in:
            self.input = ConstantInput(self.channels[self.size_in])
        self.conv1 = SpatialStyledConv(
            self.override_c_in or self.channels[self.size_in], self.channels[self.size_in], 3, self.style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = SpatialToRGB(self.channels[self.size_in], self.style_dim, upsample=False, c_out=self.c_out)

        self.log_size = int(math.log(self.size, 2))
        self.log_size_in = int(math.log(self.size_in, 2))
        self.c_in = self.channels[self.size_in]
        self.num_layers = (self.log_size - self.log_size_in) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[self.size_in]

        # for layer_idx in range(self.num_layers):
        #     res = self.size_in * 2 ** ((layer_idx + 1) // 2)
        #     shape = [1, 1, res, res]
        #     self.noises.register_buffer(f"noise_fixed_{layer_idx}", torch.randn(*shape))

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(self.log_size_in + 1, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                SpatialStyledConv(
                    in_channel,
                    out_channel,
                    3,
                    self.style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                SpatialStyledConv(
                    out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(SpatialToRGB(out_channel, self.style_dim, c_out=self.c_out))

            in_channel = out_channel

        self.n_latent = (self.log_size - self.log_size_in + 1) * 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            input=None,
            return_latents=False,
            return_image_only=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            return_features=False
    ):
        if not isinstance(styles, list):
            styles = [styles]

        if not self.spatial_style and not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent
            if self.spatial_style:
                latent = styles * inject_index
            elif styles[0].ndim < 3:
                latent = styles[0].unsqueeze(0).repeat(inject_index, 1, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            if self.spatial_style:
                latent = [styles[0]] * inject_index + [styles[1]] * (self.n_latent - inject_index)
            else:
                latent = styles[0].unsqueeze(0).repeat(inject_index, 1, 1)
                latent2 = styles[1].unsqueeze(0).repeat(self.n_latent - inject_index, 1, 1)

                latent = torch.cat([latent, latent2], 0)

        if input is not None:
            out = input
        else:
            out = self.input(latent)


        out = self.conv1(out, latent[0], noise=noise[0])

        self.outs = [out]
        skip = self.to_rgb1(out, latent[1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[i], noise=noise1)
            self.outs.append(out)
            out = conv2(out, latent[i + 1], noise=noise2)
            self.outs.append(out)
            skip = to_rgb(out, latent[i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, out
        elif return_image_only:
            return image
        else:
            return image, None


class SpatialToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], c_out=3):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = SpatialModulatedConv2d(in_channel, c_out, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, c_out, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class SpatialStyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = SpatialModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class SpatialModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.style_dim = style_dim
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        spatial_style = isinstance(style, dict)
        if spatial_style or not self.fused:
            if spatial_style:
                # style is [N, M, D_style] --> [N, M, in_channel]
                layout = style
                style = layout['spatial_style']
                style = self.modulation(style.flatten(end_dim=1)).view(*style.shape[:-1], -1)
                # layout['features'] = style
                # style = splat_features(**layout, size=input.size(-1))['feature_grid']
                style = splat_features_from_scores(layout['scores_pyramid'][input.size(-1)], style, input.size(-1),
                                                   channels_last=False)
                if self.demodulate:
                    style = style * torch.rsqrt(style.pow(2).mean([1], keepdim=True) + 1e-8)
            else:
                style = self.modulation(style).reshape(batch, in_channel, 1, 1)

            weight = self.scale * self.weight.squeeze(0)
            input = input * style

            if self.demodulate:
                if spatial_style:
                    demod = torch.rsqrt(weight.unsqueeze(0).pow(2).sum([2, 3, 4]) + 1e-8)
                    weight = weight * demod.view(self.out_channel, 1, 1, 1)
                else:
                    w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                    dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate and not style.dim() > 2:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out
