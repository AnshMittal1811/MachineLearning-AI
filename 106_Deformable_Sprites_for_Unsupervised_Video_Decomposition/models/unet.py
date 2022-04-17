import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        n_levels=3,
        d_hidden=16,
        fac=2,
        norm_fn="batch",
        init_std=0,
    ):
        super().__init__()
        enc_dims = get_unet_dims(n_levels, d_hidden)

        self.enc = Encoder(n_channels, enc_dims, norm_fn=norm_fn, fac=fac)
        self.dec = Decoder(n_classes, enc_dims, norm_fn=norm_fn, fac=fac)
        self.in_dims = self.enc.in_dims + self.dec.in_dims
        self.out_dims = self.enc.out_dims + self.dec.out_dims

        ## init the last layer
        if init_std > 0:
            print("init last zero")
            init_normal(self.dec.outc, std=init_std)
        else:
            print("init last kaiming")
            init_kaiming(self.dec.outc)

    def forward(self, x, idx=None, ret_latents=False, **kwargs):
        z, dn_latents = self.enc(x)
        out, up_latents = self.dec(z, dn_latents, ret_latents=ret_latents)
        if ret_latents:
            dn_latents.append(z)
            return out, dn_latents, up_latents
        return out


class Encoder(nn.Module):
    def __init__(self, n_channels, dims, norm_fn="batch", fac=2):
        super().__init__()
        self.n_channels = n_channels
        self.in_dims, self.out_dims = dims[:-1], dims[1:]

        self.inc = ConvBlock(n_channels, dims[0], norm_fn=norm_fn)

        self.down_layers = nn.ModuleList([])
        for d_in, d_out in zip(self.in_dims, self.out_dims):
            print("Down layer", d_in, d_out)
            self.down_layers.append(DownSkip(d_in, d_out, norm_fn=norm_fn, fac=fac))

        print("INITIALIZING WEIGHTS")
        self.apply(init_kaiming)

    def forward(self, x):
        x = self.inc(x)
        dn_latents = []
        for layer in self.down_layers:
            dn_latents.append(x)
            x = layer(x)
        return x, dn_latents


class Decoder(nn.Module):
    def __init__(self, n_classes, dims, norm_fn="batch", fac=2):
        super().__init__()
        self.n_classes = n_classes

        dims = dims[::-1]
        d_in = dims[0]
        skip_dims = dims[1:]
        out_dims = []

        self.up_layers = nn.ModuleList([])
        for d_skip in skip_dims:
            d_out = d_skip
            print("Up layer", d_in, d_out)
            self.up_layers.append(UpSkip(d_in, d_skip, norm_fn=norm_fn, fac=fac))
            d_in = d_out + d_skip
            out_dims.append(d_in)

        self.outc = nn.Conv2d(d_in, n_classes, kernel_size=1)

        self.in_dims = dims[0:1] + out_dims[:-1]
        self.out_dims = out_dims

        print("INITIALIZING WEIGHTS")
        self.apply(init_kaiming)

    def forward(self, x, dn_latents, ret_latents=False, **kwargs):
        up_latents = []
        for layer, z in zip(self.up_layers, dn_latents[::-1]):
            x = layer(x, z, ret_latents=ret_latents)
            if ret_latents:
                up_latents.append(x)
        out = self.outc(x)
        if ret_latents:
            return out, up_latents
        return out, None


def get_unet_dims(n_levels, d_hidden):
    dims = [d_hidden * 2 ** i for i in range(n_levels)]
    dims.append(dims[-1])
    return dims


class DownSkip(nn.Module):
    def __init__(self, d_in, d_out, norm_fn="batch", nl_fn="relu", fac=2):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                d_in,
                d_out,
                kernel_size=3,
                padding=1,
                stride=1,
                norm_fn=norm_fn,
                nl_fn=nl_fn,
            ),
            nn.MaxPool2d(fac),
        )

    def forward(self, x):
        return self.block(x)


class UpSkip(nn.Module):
    def __init__(self, d_in, d_out, norm_fn="batch", nl_fn="relu", fac=2):
        super().__init__()

        # use the normal convolutions to reduce the number of channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=fac, mode="bilinear", align_corners=False),
            ConvBlock(
                d_in,
                d_out,
                kernel_size=3,
                padding=1,
                stride=1,
                norm_fn=norm_fn,
                nl_fn=nl_fn,
            ),
        )

    def forward(self, x1, x2, ret_latents=False):
        x1 = self.block(x1)
        x1 = pad_diff(x1, x2)
        x1 = torch.cat([x1, x2], dim=1)
        return x1
