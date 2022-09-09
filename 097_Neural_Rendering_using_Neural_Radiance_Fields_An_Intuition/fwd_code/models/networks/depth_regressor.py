import torch.nn as nn
import torch 
import torch.nn.functional as F
import math 
from models.networks.architectures import Unet, Unet_64, Unet_128
from models.networks.cost_volum_unet import Unet_Cost_Volume
from models.networks.coding import PositionalEncoding

def get_depth_regressor(opt):
    if opt.depth_regressor == "cost_volume":
        return MVS_Depth_Regressor(opt)
    elif opt.depth_regressor == "unet":
        return  Depth_Regressor(opt)

class Depth_Regressor(nn.Module):

    def __init__(self, opt):

        super().__init__()

        if opt.depth_com:
            channels_in = 4
        else:
            channels_in = 3
        if opt.est_opacity:
            channels_out = 2
        else:
            channels_out = 1

        if opt.regressor_model=="Unet":
            self.model = Unet(channels_in=channels_in, channels_out=channels_out, opt=opt)
            self.pad = 256
        elif opt.regressor_model == "Unet64":
            self.model = Unet_64(channels_in=channels_in, channels_out=channels_out, opt=opt)
            self.pad = 64
        elif opt.regressor_model == "Unet128":
            self.model = Unet_128(channels_in=channels_in, channels_out=channels_out, opt=opt)
            self.pad = 128
        self.opt = opt

    def forward(self, depth_input, input_RTs=None,  K=None):

        B, C, H, W = depth_input.shape
        if H % self.pad == 0 and W % self.pad == 0:
            regressed_depth = self.model(depth_input)
        else:
            padding_H = math.ceil( H / self.pad) * self.pad
            padding_W = math.ceil( W / self.pad) * self.pad
            padding = max(padding_H, padding_W)
            depth_input = F.pad(depth_input, (0, padding - W, 0, padding - H), mode='constant', value=0)

            regressed_depth = self.model(depth_input)

            regressed_depth = regressed_depth[:, :, 0:H, 0:W]

        output = nn.Sigmoid()(regressed_depth)
        regressed_pts = output[:, 0:1]
        if self.opt.est_opacity:
            opacity = output[:, 1:2]
        else:
            opacity = None
        if self.opt.inverse_depth_com:
            if self.opt.normalize_depth:
                regressed_pts = regressed_pts * ( 1.0 / self.opt.min_z)
            refine_depth = regressed_pts.detach()
            regressed_pts = 1. / torch.clamp(regressed_pts, min=0.001)
        else:
            regressed_pts = (
                regressed_pts
                * (self.opt.max_z - self.opt.min_z)
                + self.opt.min_z
            )
            refine_depth = regressed_pts.detach()
        return regressed_pts, opacity, refine_depth


class MVS_Depth_Regressor(nn.Module):

    def  __init__(self,opt):

        super().__init__()
        
        if opt.depth_com:
            channels_in = 4
        else:
            channels_in = 3
        if opt.est_opacity:
            channels_out = 2
        else:
            channels_out = 1
        self.model = Unet_Cost_Volume(channels_in=channels_in, channels_out=channels_out, opt=opt)
        self.pad = 16
        self.opt = opt
        self.channels_out = channels_out

    def forward(self, input_img, input_RTs,  K, lindisp=False):
        """
        input_img: B x nv x C x H x W
        input_RTs: 
        """
        # prepare the project  matrics
        proj_matrics = input_RTs.clone()
        proj_matrics = torch.matmul(K.unsqueeze(1), proj_matrics)

        B, nv, C, H, W = input_img.shape
        depth_values = torch.linspace(0, 1, self.opt.depth_interval, device=input_img.device)
        near, far = self.opt.min_z, self.opt.max_z
        depth_values = near * (1.- depth_values ) + far * (depth_values)
        depth_values = depth_values.repeat(B, 1)
        if H % self.pad == 0 and W % self.pad == 0:
            output = self.model(input_img, proj_matrics, depth_values).view(B, nv, self.channels_out, H, W)
        else:
            padding_H = math.ceil( H / self.pad) * self.pad
            padding_W = math.ceil( W / self.pad) * self.pad
            padding = max(padding_H, padding_W)
            input_img = F.pad(input_img, (0, padding - W, 0, padding - H), mode='constant', value=0)
            output = self.model(input_img, proj_matrics, depth_values)[:, :, 0:H, 0:W].view(B, nv, self.channels_out, H, W)
        
        output = nn.Sigmoid()(output)
        regressed_pts = output[:, :, 0:1]
        if self.opt.est_opacity:
            opacity = output[:, :, 1:2]
        else:
            opacity = None
        regressed_pts = regressed_pts * (self.opt.max_z - self.opt.min_z) + self.opt.min_z
        refine_depth = regressed_pts.detach()
        return regressed_pts, opacity, refine_depth
        
