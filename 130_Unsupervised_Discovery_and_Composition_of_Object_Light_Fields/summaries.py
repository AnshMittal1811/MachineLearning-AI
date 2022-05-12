import matplotlib
matplotlib.use('Agg')

import diff_operators
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

import matplotlib.pyplot as plt

from einops import repeat

import util
import torchvision
import os
import time

from einops import rearrange

from pdb import set_trace as pdb
from torchvision.utils import save_image

def tmp(model_out, gt, writer, iter_):
    rgb(model_out, gt, writer, iter_)
    seg_vid(model_out, gt, writer, iter_)
    slot_attn_vid(model_out, gt, writer, iter_)

def rgb(model_out, gt, writer, iter_,VAL=""):

    for rgb, suffix in ((gt["query"]["rgb"],"GT"),(model_out["rgb"],"PRED"),
                        *[(rgb,f"SLOT_{i}_PRED") for i,rgb in enumerate(model_out["rgbs"])]
                       ):

        img = rearrange(rgb,"b q n c -> (b q) c n")*.5+.5

        write_img(img, writer, iter_, VAL+f"RGB_{suffix}",nrow=model_out["rgb"].size(1),
                normalize=True)

def slot_attn_vid(model_out, gt, writer, iter_,VAL=""):

    n_phi = model_out["attn"].size(1)

    attn = model_out["attn"]
    attn  = attn.permute(1,0,2).unsqueeze(2).permute(0,1,3,2).repeat(1,1,1,3)
    rgb   = gt["context"]["rgb"].flatten(0,1).unsqueeze(0).expand(n_phi,-1,-1,-1)*.5+.5
    applied_attn = attn*rgb
    tmp = torch.stack((attn,applied_attn),1)
    tmp = tmp.permute(2,0,1,3,4).flatten(0,2).permute(0,2,1)
    write_img(tmp, writer, iter_, VAL+"attn", nrow=2*n_phi, normalize=False)

def seg_vid(model_out, gt, writer, iter_,VAL=""):
    gt=gt["query"]

    soft_seg,rgb = model_out["seg"],model_out["rgbs"]

    n_phi = soft_seg.size(0)
    soft_seg = soft_seg.repeat(1,1,1,1,3)
    for i in range(n_phi):
        seg = soft_seg[i].flatten(0,1)
        img=torch.stack((seg,rgb[i].flatten(0,1)*seg),1).flatten(0,1).permute(0,2,1)
        write_img(img, writer, iter_, VAL+f"Slot_{i}_seg",nrow=2*soft_seg.size(2))

def write_img(imgs,writer,iter_,title,nrow=8,write_local=True,normalize=True):

    img_sl   = int(imgs.size(-1)**(1/2))
    img_grid = torchvision.utils.make_grid(imgs.unflatten(-1,(img_sl,img_sl)),
            scale_each=False, normalize=normalize,nrow=nrow).cpu().detach().numpy()
    if writer is None and write_local:
        plt.imshow(img_grid.transpose(1,2,0))
        plt.axis('off')
        plt.savefig(f"/home/camsmith/img/{title}.png",pad_inches=0,bbox_inches='tight')
        plt.close()
    elif writer is not None:
        writer.add_image(title, img_grid, iter_)



