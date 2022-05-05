"""
Rethinking Portrait Matting with Privacy Preserving

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting.git
Paper link: https://arxiv.org/abs/2203.16828

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import get_masked_local_from_global
from ..modules import *


class SwinStemPooling5TransformerDecoderV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ##########################
        ### Decoder part - GLOBAL
        ##########################
        self.decoder4_g = nn.Sequential(  # 768 -> 384
            nn.Conv2d(768,384,3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        self.decoder3_g = nn.Sequential(  # 384 -> 192
            nn.Conv2d(384,192,3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        self.decoder2_g = nn.Sequential(  # 192 -> 96
            nn.Conv2d(192,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder1_g = nn.Sequential(  # 96 -> 96
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder0_g = nn.Sequential(  # 96 -> 48
            nn.Conv2d(96,48,3,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,48,3,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,48,3,padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder_final_g = nn.Conv2d(48, 3, kernel_size=3, padding=1)  # 48 -> 3
        
        ##########################
        ### Decoder part - LOCAL
        ##########################

        self.decoder4_l = nn.Sequential(  # 768 -> 384
            nn.Conv2d(768,384,3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.decoder3_l = nn.Sequential(  # 384 -> 192
            nn.Conv2d(384,192,3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True))
        self.decoder2_l = nn.Sequential(  # 192 -> 96
            nn.Conv2d(192,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True))
        self.decoder1_l = nn.Sequential(  # 96 -> 96
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True))
        self.decoder0_l = nn.Sequential(  # 96 -> 48
            nn.Conv2d(96,48,3,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,48,3,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.decoder_final_l = nn.Conv2d(48,1,3,padding=1)  # 16 -> 1

        ##########################
        ### Decoder part - MODULES
        ##########################

        self.tfi_3 = TFI(384)
        self.tfi_2 = TFI(192)
        self.tfi_1 = TFI(96)
        self.tfi_0 = TFI(96)

        self.sbfi_2 = SBFI(192, 48, 8)
        self.sbfi_1 = SBFI(96, 48, 4)
        self.sbfi_0 = SBFI(96, 48, 2)

        self.dbfi_2 = DBFI(192, 768, 4)
        self.dbfi_1 = DBFI(96, 768, 8)
        self.dbfi_0 = DBFI(96, 768, 16)
    
    def forward(self, x, indices, feas):
        r"""
        x: [, 768, 16, 16]

        indices:
            [None]
            [None]
            [, 96, 64, 64]
            [, 192, 32, 32]
            [, 384, 16, 16]

        feas:
            [, 3, 512, 512]
            [, 48, 256, 256]
            [, 96, 128, 128]
            [, 192, 64, 64]
            [, 384, 32, 32]
        """
        ###########################
        ### Decoder part - Global
        ###########################
        d4_g = self.decoder4_g(x)  # 768 -> 384
        d3_g = self.decoder3_g(d4_g)  # 384 -> 192
        d2_g, global_sigmoid_side2 = self.dbfi_2(d3_g, x)  # 192, 768 -> 192
        d2_g = self.decoder2_g(d2_g)  # 192 -> 96
        d1_g, global_sigmoid_side1 = self.dbfi_1(d2_g, x)  # 96, 768 -> 96
        d1_g = self.decoder1_g(d1_g)  # 96 -> 96
        d0_g, global_sigmoid_side0 = self.dbfi_0(d1_g, x)  # 96, 768 -> 48
        d0_g = self.decoder0_g(d0_g)  # 96 -> 48
        global_sigmoid = self.decoder_final_g(d0_g)  # 48 -> 3
        ###########################
        ### Decoder part - Local
        ###########################
        d4_l = self.decoder4_l(x)  # 768 -> 384
        d4_l = F.max_unpool2d(d4_l, indices[-1], kernel_size=2, stride=2)  # 384
        d3_l = self.tfi_3(d4_g, d4_l, feas[-1])  #  384, 384, 384 -> 384
        d3_l = self.decoder3_l(d3_l)  # 384 -> 192
        d3_l = F.max_unpool2d(d3_l, indices[-2], kernel_size=2, stride=2)  # 192
        d2_l = self.tfi_2(d3_g, d3_l, feas[-2])  #  192, 192, 192 -> 192
        d2_l = self.sbfi_2(d2_l, feas[-5])  # 192, 3 -> 192
        d2_l = self.decoder2_l(d2_l)  # 192 -> 96
        d2_l  = F.max_unpool2d(d2_l, indices[-3], kernel_size=2, stride=2)  # 96
        d1_l = self.tfi_1(d2_g, d2_l, feas[-3])  #  96, 96, 96 -> 96
        d1_l = self.sbfi_1(d1_l, feas[-5])  # 96, 3 -> 96
        d1_l = self.decoder1_l(d1_l)  # 96 -> 96
        d1_l = F.max_unpool2d(d1_l, indices[-4], kernel_size=2, stride=2)  # 96
        d0_l = self.tfi_0(d1_g, d1_l, feas[-4])  #  96, 96, 96 -> 96
        d0_l = self.sbfi_0(d0_l, feas[-5])  # 96
        d0_l = self.decoder0_l(d0_l)  # 96 -> 48
        d0_l = F.max_unpool2d(d0_l, indices[-5], kernel_size=2, stride=2)  # 48
        d0_l = self.decoder_final_l(d0_l)  # 48 -> 1
        local_sigmoid = torch.sigmoid(d0_l)  # 1
        fusion_sigmoid = get_masked_local_from_global(global_sigmoid, local_sigmoid)
        return global_sigmoid, local_sigmoid, fusion_sigmoid, global_sigmoid_side2, global_sigmoid_side1, global_sigmoid_side0
