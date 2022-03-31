import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .networks import init_seq
from .diff_transforms import quaternion_to_rotation_matrix


class TextureMixDecoder(nn.Module):
    def __init__(self, channel_count=7, texture_count=32, resolution=512):
        super(TextureMixDecoder, self).__init__()
        assert channel_count in [3, 6, 7, 10]  # color, normal, roughness, specular

        self.texture_count = texture_count
        self.channel_count = channel_count
        self.resolution = resolution

        self.textures = nn.Parameter(torch.randn((texture_count, channel_count, resolution, resolution)))

    def forward(self, texcoord, mix_weights):
        '''
        Args:
            texcoord: :math:`(...,2)`
            mix_weights: :math:`(...,texture_count)`
        '''
        assert texcoord.shape[-1] == 2
        assert mix_weights.shape[-1] == self.texture_count
        assert texcoord.shape[:-1] == mix_weights.shape[:-1]

        in_shape = texcoord.shape

        sampled = F.grid_sample(self.textures.view((1, -1) + self.textures.shape[-2:]),
                                texcoord.view(1, -1, 1, 2)).permute(0, 2, 3, 1)
        sampled = sampled.view((-1, ) + self.textures.shape[:2])

        result = (sampled * mix_weights.view(-1, self.texture_count, 1)).sum(-2)

        return result.view(in_shape[:-1] + (-1, ))


class ConvTextureDecoder(nn.Module):
    def __init__(self, embedding_size=256, channel_count=7, resolution=512):
        super(ConvTextureDecoder, self).__init__()
        assert channel_count in [3, 6, 7, 10]  # color, normal, roughness, specular

        self.channel_count = channel_count
        self.resolution = resolution
        self.embedding_size = embedding_size

        in_channels = embedding_size
        out_channels = in_channels // 2

        conv_block = []
        for i in range(int(np.log2(resolution)) - 1):
            conv_block.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            conv_block.append(nn.LeakyReLU(0.2))

            if in_channels == out_channels:
                out_channels = in_channels // 2
            else:
                in_channels = out_channels
        conv_block.append(nn.ConvTranspose2d(in_channels, self.channel_count, 4, 2, 1))

        self.conv_block = nn.Sequential(*conv_block)
        init_seq(conv_block)

    def forward(self, texcoord, encoding):
        assert texcoord.shape[-1] == 2
        assert encoding.shape[-1] == self.embedding_size
        assert texcoord.shape[:-1] == encoding.shape[:-1]

        in_shape = texcoord.shape

        grid = self.conv_block(encoding.view(-1, encoding.shape[-1], 1, 1))
        sample = F.grid_sample(grid, texcoord.view(-1, 1, 1, 2))

        return sample.view(in_shape[:-1] + (sample.shape[1], ))


if __name__ == '__main__':
    M = TextureMixDecoder(7, 32, 512)
    texcoords = torch.zeros([10, 2])
    mix_weights = torch.zeros([10, 32])
    assert M(texcoords, mix_weights).shape == (10, 7)

    M = ConvTextureDecoder(256, 7, 512)
    texcoords = torch.zeros([10, 2])
    encoding = torch.zeros([10, 256])
    assert M(texcoords, encoding).shape == (10, 7)
