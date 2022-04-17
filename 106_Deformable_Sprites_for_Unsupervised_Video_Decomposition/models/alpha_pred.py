import numpy as np
import torch
import torch.nn as nn

from . import blocks
from .unet import UNet

import sys

sys.path.append("..")
import utils


class AlphaModel(nn.Module):
    def __init__(self, n_layers, net_args, d_in=3, fg_prob=0.1, **kwargs):

        super().__init__()
        assert n_layers > 1
        self.n_outputs = n_layers - 1
        self.n_layers = n_layers

        ## recenter the nonlinearity such predicted 0 maps to fg_prob
        ##  1 / (1 + exp(-(x - shift))) = fg_prob
        ## x = 0 --> shift = log(1/fg_prob - 1)
        bg_shift = np.log(1 / fg_prob - 1)
        n_remaining = torch.arange(1, self.n_outputs).float()
        fg_shift = np.log(2 * n_remaining - 1)
        shift = torch.cat([fg_shift, torch.ones(1) * bg_shift])
        self.register_buffer("shift", shift.view(1, -1, 1, 1))
        print("shifting zeros to", self.shift)

        self.backbone = UNet(d_in, self.n_outputs, **net_args)

    def forward(self, x, **kwargs):
        """
        :param x (N, C, H, W)
        """
        pred = self.backbone(x, **kwargs)  # (N, M-1, H, W)
        return self._pred_to_output(pred)

    def _pred_to_output(self, x):
        """
        turn model output into layer weights
        :param x (B, M-1, H, W)
        """
        ## predict occupancies
        x = x - self.shift
        x = torch.sigmoid(x).unsqueeze(2)  # (B, M-1, 1, H, W)

        ## we predict the complement of bg to initialize all weights near 0
        fg = x[:, :-1]  # (B, M-2, ...)
        bg = 1 - x[:, -1:]  # (B, 1, ...)
        occ = torch.cat(
            [torch.zeros_like(x[:, :1]), fg, torch.ones_like(x[:, :1])], dim=1
        )

        ## compute visibility from back to front
        ## v(1) = 1
        ## v(2) = (1 - o(1))
        ## v(M-1) = (1 - o(1)) * ... * (1 - o(M-2))
        ## v(M) = 0
        vis = torch.cumprod(1 - occ, dim=1)
        acc = vis[:, :-1] * occ[:, 1:]  # (B, M-1, ...)
        weights = torch.cat([(1 - bg) * acc, bg], dim=1)

        return {"masks": weights, "alpha": acc, "pred": x}
