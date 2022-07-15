import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import pano_upsample_w, PanoUpsampleW


'''
Simple decoder (for s2d3d sem small input size)
'''
class SimpleReduction(nn.Module):
    def __init__(self, cs, heights, out_ch=64):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(SimpleReduction, self).__init__()
        c1, c2, c3, c4 = cs
        h1, h2, h3, h4 = heights

        def EfficientConvCompressH(in_c, out_c, scale, down_h):
            return nn.Sequential(
                PanoUpsampleW(scale),
                nn.Conv2d(in_c, out_c, (down_h, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.ghc_lst = nn.ModuleList([
            EfficientConvCompressH(c1, c1//4, scale=1, down_h=h1),
            EfficientConvCompressH(c2, c2//4, scale=2, down_h=h2),
            EfficientConvCompressH(c3, c3//4, scale=4, down_h=h3),
            EfficientConvCompressH(c4, c4//4, scale=8, down_h=h4),
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d((c1+c2+c3+c4)//4, out_ch, (1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_ch

    def forward(self, conv_list):
        assert len(conv_list) == 4
        feature = torch.cat([
            f(x) for f, x in zip(self.ghc_lst, conv_list)
        ], dim=1)
        feature = self.fuse(feature).squeeze(2)
        return {'1D': feature}
