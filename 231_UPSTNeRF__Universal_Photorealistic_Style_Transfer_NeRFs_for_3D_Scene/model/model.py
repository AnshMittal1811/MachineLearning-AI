import torch
import torch.nn as nn
import os
import logging
import torchvision

from collections import OrderedDict
from model.nerf import *
from model import RAIN
from model.RAIN import Net as RAIN_net

logger = logging.getLogger(__package__)


class StyleNeRFpp(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.stage = args.stage
        
        self.nerf_net = NerfNet(args)
        
        self.latent_codes = nn.Embedding(1, 64).cuda()
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
		
        if args.stage != "first":
            # Create vgg and fc_encoder in RAIN_net
            vgg = RAIN.vgg
            fc_encoder = RAIN.fc_encoder

            # Load pretrained weights of vgg and fc_encoder
            vgg.load_state_dict(torch.load(args.vgg_pretrained_path))
            fc_encoder.load_state_dict(torch.load(args.fc_encoder_pretrained_path))

            vgg = nn.Sequential(*list(vgg.children())[:31])
            self.RAIN_net = RAIN_net(vgg, fc_encoder)

            # Fixed RAIN_net
            for param in self.RAIN_net.parameters():
                param.requires_grad = False
    
    def get_content_feat(self, content_img):
        return self.RAIN_net.get_content_feat(content_img)
    
    def get_style_feat(self, style_img):
        return self.RAIN_net.get_style_feat(style_img)

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, style_img, style_idx):#ray_o([5427, 3]);style_img([3, 256, 256])
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        latent = self.latent_codes(torch.tensor(0).cuda().long())#64
        
        if self.stage != "first":
            style_mean = self.RAIN_net.get_hyper_input(style_img.cuda().unsqueeze(0))#style_img([3, 256, 256])->style_mean([1, 512])
            style_latent = style_mean.clone().detach()
        else:
            style_latent = None
        
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, latent, style_latent)

        return ret
