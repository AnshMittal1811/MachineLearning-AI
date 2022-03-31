import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .networks import init_seq, positional_encoding
from .base_decoder import BaseDecoder

class RadianceDecoder(BaseDecoder):
    def __init__(self, num_freqs, num_viewdir_freqs):
        super(RadianceDecoder, self).__init__()

        self.num_freqs = num_freqs if num_freqs > 0 else 0
        self.num_viewdir_freqs = num_viewdir_freqs if num_viewdir_freqs > 0 else 0

        self.out_channels = 3
        self.in_channels = 2 * self.num_freqs * 3

        self.viewdir_channels = 2 * self.num_viewdir_freqs * 3

        out_channels = 256

        in_channels = self.in_channels
        out_channels = 256
        block1 = []
        for i in range(5):
            block1.append(nn.Linear(in_channels, out_channels))
            block1.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = 256
        self.block1 = nn.Sequential(*block1)

        block2 = []
        in_channels = in_channels + self.in_channels
        out_channels = 256
        for i in range(4):
            block2.append(nn.Linear(in_channels, out_channels))
            block2.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = 256

        self.block2 = nn.Sequential(*block2)

        self.alpha_branch = nn.Sequential(nn.Linear(in_channels, 1), nn.ReLU())

        block3 = []
        in_channels = in_channels + self.viewdir_channels
        out_channels = 128
        for i in range(4):
            block3.append(nn.Linear(in_channels, out_channels))
            block3.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = 128

        block3.append(nn.Linear(in_channels, self.out_channels))
        self.block3 = nn.Sequential(*block3)

        for m in [self.block0, self.block1, self.block2, self.alpha_branch, self.block3]:
            init_seq(m)

    def forward(self, pts, viewdirs, encoding=None, **kwargs):
        # input:
        # pts: [..., 3]
        # viewdirs: [..., 3], must be normalized
        if encoding is not None:
            encoding = encoding[:, None, None, :].expand(pts.shape[:3] + (encoding.shape[-1], ))
            encoding = encoding.view(-1, encoding.shape[-1])

        in_shape = pts.shape

        if self.num_freqs > 0:
            freq_bands = (2**np.arange(self.num_freqs).astype(np.float32))  #* np.pi
            pts = torch.cat([(pts * freq) for freq in freq_bands], dim=-1)  # ... x (3*num_freqs)
            pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)  # (... x (3*num_freqs*2))
        pts = pts.view((-1, pts.shape[-1]))  #(M, 3*num_freqs*2)

        if self.num_viewdir_freqs > 0:
            freq_bands = (2**np.arange(self.num_viewdir_freqs).astype(np.float32))  #* np.pi
            viewdirs = torch.cat([(viewdirs * freq) for freq in freq_bands], dim=-1)  # ... x (3*num_freqs)
            viewdirs = torch.cat([torch.sin(viewdirs), torch.cos(viewdirs)],
                                 dim=-1)  # (... x (3*num_freqs*2))
        viewdirs = viewdirs.view((-1, viewdirs.shape[-1]))  #(M, 3*num_freqs*2)

        output = self.block1(pts)
        output = torch.cat([output, pts], dim=-1)
        if self.use_encoding:
            assert encoding is not None
            output = torch.cat([output, encoding], dim=-1)
        output = self.block2(output)

        alpha = self.alpha_branch(output)
        alpha = alpha.view(in_shape[:-1] + (-1, ))

        output = torch.cat([output, viewdirs], dim=-1)
        output = self.block3(output)
        output = torch.sigmoid(output)
        #  output = output.view((in_pts.shape[0], in_pts.shape[1], -1))
        output = output.view(in_shape[:-1] + (-1, ))

        output = torch.cat([alpha, output], dim=-1)

        return output


class MlpDecoder(BaseDecoder):
    def __init__(self, num_freqs, out_channels=8, encoding_size=0, encoding_freqs=6):
        super(MlpDecoder, self).__init__()
        self.num_freqs = num_freqs if num_freqs > 0 else 0
        self.encoding_freqs = encoding_freqs

        self.out_channels = out_channels
        self.position_in_channels = 2 * self.num_freqs * 3

        out_channels = 256
        if encoding_size < 0:
            encoding_size = 0
        self.use_encoding = encoding_size > 0
        if self.use_encoding:
            in_channels = 2 * encoding_size * encoding_freqs
            out_channels = 256
            block0 = []
            for i in range(3):
                block0.append(nn.Linear(in_channels, out_channels))
                block0.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                out_channels = 256
            self.block0 = nn.Sequential(*block0)

        # block 1
        in_channels = self.position_in_channels
        if self.use_encoding:
            in_channels += 256
        out_channels = 256
        block1 = []
        for i in range(5):
            block1.append(nn.Linear(in_channels, out_channels))
            block1.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = 256
        self.block1 = nn.Sequential(*block1)

        block2 = []
        out_channels = 256
        for i in range(4):
            block2.append(nn.Linear(in_channels, out_channels))
            block2.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = 256

        out_channels = 128
        for i in range(4):
            block2.append(nn.Linear(in_channels, out_channels))
            block2.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = 128

        block2.append(nn.Linear(in_channels, self.out_channels))
        self.block2 = nn.Sequential(*block2)

        for m in [self.block0, self.block1, self.block2]:
            init_seq(m)

    def forward(self, pts, viewdirs=None, encoding=None, **kwargs):
        in_shape = pts.shape

        if self.use_encoding:
            assert encoding is not None
        if self.use_encoding:
            encoding = encoding[:, None, None, :].expand(pts.shape[:3] + (encoding.shape[-1], ))
            encoding = encoding.reshape((-1, encoding.shape[-1]))
            pe0 = positional_encoding(encoding, self.encoding_freqs)
            output0 = self.block0(pe0)

        if self.num_freqs:
            pe1 = positional_encoding(pts, self.num_freqs)
            # freq_bands = (2**np.arange(self.num_freqs).astype(np.float32)) * np.pi
            # pts = torch.cat([(pts * freq) for freq in freq_bands], dim=-1)  # ... x (3*num_freqs)
            # pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)  # ... x (3*num_freqs*2)
        pe1 = pe1.view((-1, pe1.shape[-1]))  #(N, 3*num_freqs*2)

        pts = pe1
        if self.use_encoding:
            pts = torch.cat([output0, pe1], dim=-1)

        output = self.block1(pts)
        output = self.block2(output)
        output = output.view(in_shape[:-1] + (-1, ))

        # sigma
        output[..., [0]] = F.softplus(output[..., [0]])

        # albedo
        if self.out_channels >= 4:
            output[..., 1:4] = torch.sigmoid(output[..., 1:4])

        # normal
        if self.out_channels >= 7:
            output[..., 4:7] = torch.sigmoid(output[..., 4:7])
            output[..., 4:7] = F.normalize(2.0 * output[..., 4:7] - 1.0, dim=-1)

        # roughness
        if self.out_channels >= 8:
            output[..., 7] = torch.sigmoid(output[..., 7])

        # specular albedo
        if self.out_channels >= 11:
            output[..., 8:11] = torch.sigmoid(output[..., 8:11])  # specular albedo

        return output
