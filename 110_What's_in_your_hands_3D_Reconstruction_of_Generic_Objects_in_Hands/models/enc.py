import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34
import pytorch_lightning as pl


class ImageSpEnc(pl.LightningModule):
    def __init__(self, cfg, out_dim, inp_dim=3, modality='rgb', *args, **kwargs):
        super().__init__()
        # self.hparams = cfg
        self.hparams.update(cfg)
        self.cfg = cfg
        self.modality = modality

        model = resnet34(True)
        if modality in ['rgb', 'gray', 'fg']:
            pass
        elif modality == 'rgba':
            base = nn.Conv2d(4, 64, 7, 2, 3, bias=False)         # 64, 3 ,7, 7
            base_weight = model.conv1.weight
            base.weight[:, 0:3].data.copy_(base_weight)
            base.weight[:, 3].data.copy_(torch.mean(base_weight, dim=1))
            
            model.conv1 = base
        elif modality == 'flow':
            base = nn.Conv2d(6, 64, 7, 2, 3, bias=False)         # 64, 3 ,7, 7
            base_weight = model.conv1.weight
            base.weight[:, 0:3].data.copy_(base_weight)
            base.weight[:, 3:6].data.copy_(base_weight)

            model.conv1 = base
        else:
            raise NotImplementedError(modality)
        # 224 --> 56 -> 28 -> 14 -> 7
        self.net = model

        # dim = 256+512+1024+2048
        dim = 64+128+256+512
        self.z_head = nn.Conv2d(dim, out_dim, 1)
        self.global_head = nn.Sequential(nn.Conv2d(512, out_dim, 1), nn.Flatten())
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # change enc channel
        if self.modality == 'rgba' and state_dict['enc.net.conv1.weight'].size(1) == 3:
            base = state_dict['enc.net.conv1.weight'].clone()
            mean = torch.mean(base, dim=1, keepdim=True)
            weight = torch.cat([base, mean], 1)
            state_dict['enc.net.conv1.weight'] = weight

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _foward_res(self, x):
        latent =[]

        net = self.net
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        latent.append(x)
        x = net.layer2(x)
        latent.append(x)
        x = net.layer3(x)
        latent.append(x)
        x = net.layer4(x)
        latent.append(x)
        x = net.avgpool(x)
        
        return latent, x

    def forward(self, x, mask=None, flow=None):
        if self.modality == 'rgb':
            x = x 
        elif self.modality == 'rgba':
            x = torch.cat([x, mask], dim=1)
        elif self.modality == 'flow':
            x = torch.cat([x, flow], dim=1)
        elif self.modality == 'gray':
            x = x * mask
            x = torch.mean(x, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        elif self.modality == 'fg':
            x = x * mask
        else:
            raise NotImplementedError

        latents, glb_latents = self._foward_res(x)
        align_corners = True
        latent_sz = latents[0].shape[-2:]
        for i, lat in enumerate(latents):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode='bilinear',
                align_corners=align_corners,
            )
        latents = torch.cat(latents, dim=1)
        latents = self.z_head(latents)
        glb_latents = self.global_head(glb_latents)

        return glb_latents, latents


def build_net(name, cfg) -> ImageSpEnc:
    return ImageSpEnc(cfg,  out_dim=cfg.MODEL.Z_DIM, layer=cfg.MODEL.ENC_RESO, modality=cfg.DB.INPUT)
