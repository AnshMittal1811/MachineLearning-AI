import torch.nn as nn
import torch 
import torch.nn.functional as F
import math 
from models.networks.architectures import (
    ResNetDecoder,
    UNetDecoder64,
)

class Decoder(nn.Module):

    def __init__(self, opt, norm="batch"):
        super().__init__()
        decode_in_dim = opt.decode_in_dim 
        if opt.refine_model_type == "unet":
            self.model = UNetDecoder64(opt, channels_in=decode_in_dim, channels_out=3)
        elif "resnet" in opt.refine_model_type:
            print("RESNET decoder")
            self.model = ResNetDecoder(opt, channels_in=decode_in_dim, channels_out=3, norm=norm)
    def forward(self, gen_fs, scale_factor=1.0):
        factor = 2
        B, C, H, W = gen_fs.shape
        if H %factor == 0 and W % factor == 0:
            gen_img =  self.model(gen_fs, scale_factor)
            return gen_img
        else:
            padding_H = math.ceil( H / factor) * factor
            padding_W = math.ceil( W / factor) * factor
            gen_fs = F.pad(gen_fs, (0, padding_W - W, 0, padding_H - H), mode='constant', value=0)
            gen_img = self.model(gen_fs, scale_factor)

            return gen_img[:, :, 0:H, 0:W]