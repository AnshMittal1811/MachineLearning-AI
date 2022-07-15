import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import pano_upsample_w, PanoUpsampleW


'''
Original HC
'''
class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()

        def ConvCompressH(in_c, out_c, ks=3):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)
        assert out_w % x.shape[3] == 0
        return pano_upsample_w(x, out_w//x.shape[-1])


class GlobalHeightStage(nn.Module):
    def __init__(self, cs, heights, down_h=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        c1, c2, c3, c4 = cs
        h1, h2, h3, h4 = heights
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//down_h),
            GlobalHeightConv(c2, c2//down_h),
            GlobalHeightConv(c3, c3//down_h),
            GlobalHeightConv(c4, c4//down_h),
        ])
        self.out_channels = (c1*h1 + c2*h2 + c3*h3 + c4*h4) // 16 // down_h

    def forward(self, conv_list):
        assert len(conv_list) == 4
        bs, _, _, out_w = conv_list[0].shape
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x in zip(self.ghc_lst, conv_list)
        ], dim=1)
        return {'1D': feature}
