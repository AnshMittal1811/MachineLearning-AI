import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..networks import init_seq


class ConvTemplate(nn.Module):
    def __init__(self, input_dim, out_channels, template_res, size=1024):
        super().__init__()

        self.size = size
        self.input_dim = input_dim
        self.out_channels = out_channels
        self.template_res = template_res

        self.template1 = nn.Sequential(
            nn.Linear(self.input_dim, size), nn.LeakyReLU(0.2)
        )

        template2 = []
        in_channels, out_channels = size, size // 2
        for i in range(int(np.log2(self.template_res)) - 1):
            template2.append(nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1))
            template2.append(nn.LeakyReLU(0.2))
            if in_channels == out_channels:
                out_channels = in_channels // 2
            else:
                in_channels = out_channels
        template2.append(nn.ConvTranspose3d(in_channels, self.out_channels, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)

        for m in [self.template1, self.template2]:
            init_seq(m)

    def forward(self, input_encoding, pts):
        """
        Args
            encoding: input encoding vector, :math:`(N, encoding_size)`
            pts: input points, :math:`(N, ... , 3)`
        Return
            output: point features :math:`(N, ..., channels)`
        """
        template = self.template2(
            self.template1(input_encoding).view(-1, self.size, 1, 1, 1)
        )
        output = F.grid_sample(
            template,
            pts.view((pts.shape[0], -1, 1, 1, 3)),
            padding_mode="border",
            align_corners=True,
        )
        output = output.permute(0, 2, 3, 4, 1)
        return output.view(pts.shape[:-1] + (self.out_channels,))


class SmallConvTemplate(nn.Module):
    def __init__(self, input_dim, out_channels, template_res):
        super().__init__()

        self.size = 128
        self.input_dim = input_dim
        self.out_channels = out_channels
        self.template_res = template_res

        self.template1 = nn.Sequential(
            nn.Linear(self.input_dim, self.size), nn.LeakyReLU(0.2)
        )

        template2 = []
        in_channels, out_channels = self.size, self.size
        for i in range(int(np.log2(self.template_res)) - 1):
            template2.append(nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1))
            template2.append(nn.LeakyReLU(0.2))
        template2.append(nn.ConvTranspose3d(out_channels, self.out_channels, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)

        for m in [self.template1, self.template2]:
            init_seq(m)

    def forward(self, input_encoding, pts):
        """
        Args
            encoding: input encoding vector, :math:`(N, encoding_size)`
            pts: input points, :math:`(N, ... , 3)`
        Return
            output: point features :math:`(N, ..., channels)`
        """
        template = self.template2(
            self.template1(input_encoding).view(-1, self.size, 1, 1, 1)
        )
        output = F.grid_sample(
            template,
            pts.view((pts.shape[0], -1, 1, 1, 3)),
            padding_mode="border",
            align_corners=True,
        )
        output = output.permute(0, 2, 3, 4, 1)
        return output.view(pts.shape[:-1] + (self.out_channels,))


class SmallConvTemplate2D(nn.Module):
    def __init__(self, input_dim, out_channels, template_res):
        super().__init__()

        self.size = 128
        self.input_dim = input_dim
        self.out_channels = out_channels
        self.template_res = template_res

        self.template1 = nn.Sequential(
            nn.Linear(self.input_dim, self.size), nn.LeakyReLU(0.2)
        )

        template2 = []
        in_channels, out_channels = self.size, self.size
        for i in range(int(np.log2(self.template_res)) - 1):
            template2.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            template2.append(nn.LeakyReLU(0.2))
        template2.append(nn.ConvTranspose2d(out_channels, self.out_channels, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)

        for m in [self.template1, self.template2]:
            init_seq(m)


    def forward(self, input_encoding, pts):
        """
        Args
            encoding: input encoding vector, :math:`(N, encoding_size)`
            pts: input points, :math:`(N, ... , 2)`
        Return
            output: point features :math:`(N, ..., channels)`
        """
        template = self.template2(
            self.template1(input_encoding).view(-1, self.size, 1, 1)
        )
        output = F.grid_sample(
            template,
            pts.view((pts.shape[0], -1, 1, 2)),
            padding_mode="border",
            align_corners=True,
        )
        output = output.permute(0, 2, 3, 1)
        return output.view(pts.shape[:-1] + (self.out_channels,))
