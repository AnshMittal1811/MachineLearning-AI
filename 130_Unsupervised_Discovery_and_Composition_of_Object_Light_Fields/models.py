import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import math

from collections import OrderedDict

from einops import repeat,rearrange
from einops.layers.torch import Rearrange

from pdb import set_trace as pdb #debugging

from copy import deepcopy

import torchvision
import util

import custom_layers
import geometry
import hyperlayers

import conv_modules

from torch.nn.functional import normalize

class COLF(nn.Module):

    def __init__(self,phi_latent=128, num_phi=1, phi_out_latent=64,
                hyper_hidden=1,phi_hidden=2, img_feat_dim=128,zero_bg=False):

        super().__init__()

        self.num_phi=num_phi
        self.zero_bg=zero_bg

        num_hidden_units_phi = 256

        self.phi = custom_layers.FCBlock(
                                hidden_ch=num_hidden_units_phi,
                                num_hidden_layers=phi_hidden,
                                in_features=6,
                                out_features=phi_out_latent,
                                outermost_linear=True,)
        self.hyper_fg = hyperlayers.HyperNetwork(
                              hyper_in_features=phi_latent,
                              hyper_hidden_layers=hyper_hidden,
                              hyper_hidden_features=num_hidden_units_phi,
                              hypo_module=self.phi)
        self.hyper_bg = hyperlayers.HyperNetwork(
                              hyper_in_features=phi_latent,
                              hyper_hidden_layers=hyper_hidden,
                              hyper_hidden_features=num_hidden_units_phi,
                              hypo_module=self.phi)


        # Maps pixels to features for SlotAttention
        self.img_encoder = nn.Sequential(
                conv_modules.UnetEncoder(bottom=True,z_dim=img_feat_dim),
                Rearrange("b c x y -> b (x y) c")
        )
        self.slot_encoder = custom_layers.SlotAttention(self.num_phi,
                                                       in_dim=img_feat_dim,
                                                       fg_slot_dim=phi_latent,
                                                       bg_slot_dim=phi_latent,
                                                       max_slot_dim=phi_latent)

        self.feat_to_depth  = custom_layers.FCBlock(
                        hidden_ch=128, num_hidden_layers=3, in_features=phi_out_latent,
                        out_features=1, outermost_linear=True,
                        norm='layernorm_na')
        self.depth_spreader = custom_layers.FCBlock(
                        hidden_ch=128, num_hidden_layers=3, in_features=2,
                        out_features=1, outermost_linear=True,
                        norm='layernorm_na')

        # Maps features to rgb
        self.pix_gen_bg = nn.Sequential( custom_layers.FCBlock(
                        hidden_ch=128, num_hidden_layers=3, in_features=phi_out_latent,
                        out_features=3, outermost_linear=True,
                        norm='layernorm_na'), nn.Tanh() )
        self.pix_gen_fg = nn.Sequential( custom_layers.FCBlock(
                        hidden_ch=128, num_hidden_layers=3, in_features=phi_out_latent,
                        out_features=3, outermost_linear=True,
                        norm='layernorm_na'), nn.Tanh() )
        print(self)

    def compositor(self,feats):
        depth = self.feat_to_depth(feats)
        depth = rearrange(depth,"p b q pix 1 -> (b q pix) p 1")
        min_depth = depth.min(1,keepdim=True)[0].expand(-1,feats.size(0),1)
        attn = rearrange(self.depth_spreader(torch.cat((min_depth-depth,depth),-1)),
                            "(b q pix) p 1 -> p b q pix 1",
                            p=feats.size(0),b=feats.size(1),q=feats.size(2))
        return attn.softmax(0)+1e-9

    def forward(self,input):
        query = input['query']
        b, n_ctxt = query["uv"].shape[:2]
        n_qry, n_pix = query["uv"].shape[1:3]
        cam2world, query_intrinsics, query_uv = util.get_query_cam(input)
        phi_intrinsics,phi_uv = [x.unsqueeze(0).expand(self.num_phi-1,-1,-1,-1)
                                        for x in (query_intrinsics,query_uv)]

        # Encode all imgs
        imsize = int(input["context"]["rgb"].size(-2)**(1/2))
        rgb_A = input["context"]["rgb"][:,0].permute(0,2,1).unflatten(-1,(imsize,imsize))

        # Create fg images: img_encoding -> slot attn -> compositor,rgb

        imfeats = self.img_encoder(rgb_A)
        slots, attn = self.slot_encoder(imfeats) # b phi l 

        context_cam = input["context"]["cam2world"][:,0]

        world2contextcam = repeat(context_cam.inverse(),"b x y -> (b q) x y",q=n_qry)
        fg_pose = repeat(world2contextcam @ cam2world,"bq x y -> p bq x y",p=self.num_phi-1)
        fg_coords=geometry.plucker_embedding(fg_pose,phi_uv,phi_intrinsics)
        bg_coords = geometry.plucker_embedding(cam2world,query_uv,query_intrinsics)
        coords = torch.cat((bg_coords[None],fg_coords)).flatten(0,1)

        # Create phi
        fg_rep=repeat(slots[:,1:],"b p l -> p (b q) l",q=n_qry)
        bg_rep=repeat(slots[:,:1],"b p l -> p (b q) l",q=n_qry)
        fg_params = self.hyper_fg(fg_rep)
        bg_params = self.hyper_bg(bg_rep)
        phi_params=OrderedDict()
        for k in bg_params.keys(): 
            phi_params[k]=torch.cat([bg_params[k],fg_params[k]])

        feats = self.phi(coords,params=phi_params)
        feats = rearrange(feats, "(p b q) pix l -> p b q pix l", p=self.num_phi,b=b,q=n_qry)
        rgbs = torch.cat((self.pix_gen_bg(feats[:1]),self.pix_gen_fg(feats[1:])))
        seg = self.compositor(feats)
        rgb  = (rgbs*seg).sum(0) # AB   b q pix 3

        out_dict = {
            "rgbs":rgbs,
            "rgb": rgb,
            "seg": seg,
            "attn":attn,
            "fg_latent":slots[:,1:],
        }
        return out_dict
