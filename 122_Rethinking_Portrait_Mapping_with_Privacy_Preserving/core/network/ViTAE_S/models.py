# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

from .base_model import ViTAE_noRC_MaxPooling_basic
from ..modules import TFI, SBFI, DBFI
from util import get_masked_local_from_global
from config import PRETRAINED_VITAE_NORC_MAXPOOLING_BIAS_BASIC_STAGE4_14


##########################################
## Encoder
##########################################

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ViTAE_stages3_7': _cfg(),
}

@register_model
def ViTAE_noRC_MaxPooling_bias_basic_stages4_14(pretrained=True, **kwargs): # adopt performer for tokens to token
    model = ViTAE_noRC_MaxPooling_basic(RC_tokens_type=['performer', 'transformer', 'transformer', 'transformer'], NC_tokens_type=['performer', 'transformer', 'transformer', 'transformer'], stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                            NC_depth=[2, 2, 12, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], RC_group=[1, 16, 32, 64], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_stages3_7']
    if pretrained:
        ckpt = torch.load(PRETRAINED_VITAE_NORC_MAXPOOLING_BIAS_BASIC_STAGE4_14)['state_dict_ema']
        model.load_state_dict(ckpt, strict=True)
    return model


##########################################
## Decoder
##########################################

class ViTAE_noRC_MaxPooling_DecoderV1(nn.Module):
    def __init__(self):
        super().__init__()
        ##########################
        ### Decoder part - GLOBAL
        ##########################
        in_chan = 512
        out_chan = 256
        self.decoder4_g = nn.Sequential(  # 512 -> 256
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )

        in_chan = 256
        out_chan = 128
        self.decoder3_g = nn.Sequential(  # 256 -> 128
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )

        in_chan = 128
        out_chan = 64
        self.decoder2_g = nn.Sequential(  # 128 -> 64
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))

        in_chan = 64
        out_chan = 64
        self.decoder1_g = nn.Sequential(  # 64 -> 64
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))

        in_chan = 64
        out_chan = 32
        self.decoder0_g = nn.Sequential(  # 64 -> 32
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder_final_g = nn.Conv2d(out_chan, 3, kernel_size=3, padding=1)  # 32 -> 3
        
        ##########################
        ### Decoder part - LOCAL
        ##########################

        in_chan = 512
        out_chan = 256
        self.decoder4_l = nn.Sequential(  # 512 -> 256
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))

        in_chan = 256
        out_chan = 128
        self.decoder3_l = nn.Sequential(  # 256 -> 128
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))

        in_chan = 128
        out_chan = 64
        self.decoder2_l = nn.Sequential(  # 128 -> 64
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))

        in_chan = 64
        out_chan = 64
        self.decoder1_l = nn.Sequential(  # 64 -> 64
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))
        
        in_chan = 64
        out_chan = 32
        self.decoder0_l = nn.Sequential(  # 64 -> 32
            nn.Conv2d(in_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan,out_chan,3,padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))
        self.decoder_final_l = nn.Conv2d(out_chan,1,3,padding=1)  # 32 -> 1

        ##########################
        ### Decoder part - MODULES
        ##########################

        self.tfi_3 = TFI(256)
        self.tfi_2 = TFI(128)
        self.tfi_1 = TFI(64)
        self.tfi_0 = TFI(64)

        self.sbfi_2 = SBFI(128, 32, 8)
        self.sbfi_1 = SBFI(64, 32, 4)
        self.sbfi_0 = SBFI(64, 32, 2)

        self.dbfi_2 = DBFI(128, 512, 4)
        self.dbfi_1 = DBFI(64, 512, 8)
        self.dbfi_0 = DBFI(64, 512, 16)
    
    def forward(self, x, indices, feas):
        ###########################
        ### Decoder part - Global
        ###########################
        d4_g = self.decoder4_g(x)  # 512 -> 256
        d3_g = self.decoder3_g(d4_g)  # 256 -> 128
        d2_g, global_sigmoid_side2 = self.dbfi_2(d3_g, x)  # 128, 512 -> 128
        d2_g = self.decoder2_g(d2_g)  # 128 -> 64
        d1_g, global_sigmoid_side1 = self.dbfi_1(d2_g, x)  # 64, 512 -> 64
        d1_g = self.decoder1_g(d1_g)  # 64 -> 64
        d0_g, global_sigmoid_side0 = self.dbfi_0(d1_g, x)  # 64, 512 -> 64
        d0_g = self.decoder0_g(d0_g)  # 64 -> 32
        global_sigmoid = self.decoder_final_g(d0_g)  # 32 -> 3
        ###########################
        ### Decoder part - Local
        ###########################
        d4_l = self.decoder4_l(x)  # 512 -> 256
        d4_l = F.max_unpool2d(d4_l, indices[-1], kernel_size=2, stride=2)  # 256
        d3_l = self.tfi_3(d4_g, d4_l, feas[-1])  #  256, 256, 256 -> 256
        d3_l = self.decoder3_l(d3_l)  # 256 -> 128
        d3_l = F.max_unpool2d(d3_l, indices[-2], kernel_size=2, stride=2)  # 128
        d2_l = self.tfi_2(d3_g, d3_l, feas[-2])  # 128, 128, 128 -> 128
        d2_l = self.sbfi_2(d2_l, feas[-5])  # 128, 32 -> 128
        d2_l = self.decoder2_l(d2_l)  # 128 -> 64
        d2_l  = F.max_unpool2d(d2_l, indices[-3], kernel_size=2, stride=2)  # 64
        d1_l = self.tfi_1(d2_g, d2_l, feas[-3])  #  64, 64, 64 -> 64
        d1_l = self.sbfi_1(d1_l, feas[-5])  # 64, 32 -> 64
        d1_l = self.decoder1_l(d1_l)  # 64 -> 64
        d1_l = F.max_unpool2d(d1_l, indices[-4], kernel_size=2, stride=2)  # 64
        d0_l = self.tfi_0(d1_g, d1_l, feas[-4])  #  64, 64, 64 -> 64
        d0_l = self.sbfi_0(d0_l, feas[-5])  # 64, 32 -> 64
        d0_l = self.decoder0_l(d0_l)  # 64 -> 32
        d0_l = F.max_unpool2d(d0_l, indices[-5], kernel_size=2, stride=2)  # 32
        d0_l = self.decoder_final_l(d0_l)  # 32 -> 1
        local_sigmoid = torch.sigmoid(d0_l)  # 1
        fusion_sigmoid = get_masked_local_from_global(global_sigmoid, local_sigmoid)
        return global_sigmoid, local_sigmoid, fusion_sigmoid, global_sigmoid_side2, global_sigmoid_side1, global_sigmoid_side0


##########################################
## Matting Model
##########################################

class ViTAE_noRC_MaxPooling_Matting(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        embeddings, indices, feas = self.encoder(x)
        return self.decoder(embeddings, indices, feas)
