import util
import matplotlib.pyplot as plt

import random
import numpy as np
import torch
import torch.nn.functional as F
import diff_operators
import torch.nn as nn
from einops import repeat,rearrange

from pdb import set_trace as pdb #debug

import conv2d_gradfix

import lpips

if 'loss_fn_alex' not in globals():
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def tmp(model_out,gt):
    print( rgb(model_out,gt) )
    print( latent_penalty(model_out,gt) )
    print( lpips_loss(model_out,gt) )
    return

def latent_penalty(model_out,gt):
    return model_out["fg_latent"].square().mean()

def l1_rgb(model_out,gt):
    return F.l1_loss(model_out["rgb"],gt["rgb"])

def rgb(model_out,gt):
    return F.mse_loss(model_out["rgb"],gt["rgb"])

def lpips_loss(model_out,gt):
    imsl=int(gt["rgb"].size(-2)**(1/2))
    pred_rgb,gt_rgb=[src["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
                for src in (model_out,gt)]
    return loss_fn_alex(pred_rgb,gt_rgb).mean()
