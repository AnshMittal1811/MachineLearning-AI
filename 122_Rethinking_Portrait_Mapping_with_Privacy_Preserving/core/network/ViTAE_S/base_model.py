"""
Rethinking Portrait Matting with Privacy Preserving

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting.git
Paper link: https://arxiv.org/abs/2203.16828

"""

from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
from .NormalCell import NormalCell
from timm.models.layers import to_2tuple, trunc_normal_

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(in_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, input_resolution=None):
        """
        x: B, H*W, C
        """
        if input_resolution is None:
            input_resolution = self.input_resolution
        H, W = input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x_fea = x  # 64, 128, 256
        x, idx = self.pooling(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()
        x = self.linear(x)

        return x, [idx], [x_fea]

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * self.dim * 2 * self.dim * 2
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        _patch_size = patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        strides = []
        while _patch_size > 1:
            strides.append(2)
            _patch_size = _patch_size // 2
        if len(strides) < 3:
            strides.append(1)
        self.strides = strides
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.inter_chans = embed_dim // 2
        self.embed_dim = embed_dim
        # self.proj = nn.Sequential(
        #     nn.Conv2d(self.in_chans, self.inter_chans, 3, stride=strides[0], padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(self.inter_chans, embed_dim, 3, stride=strides[1], padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, 3, stride=strides[2], padding=1)
        # )

        self.proj1 = nn.Conv2d(self.in_chans, self.inter_chans, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.proj2 = nn.Conv2d(self.inter_chans, embed_dim, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.proj3 = nn.Conv2d(embed_dim, embed_dim, 3, stride=1, padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, input_resolution=None):
        x = self.relu1(self.proj1(x))
        
        x_fea1 = x  # 32
        x, idx1 = self.maxpool1(x)

        x = self.relu2(self.proj2(x))
        x_fea2 = x  # 64
        x, idx2 = self.maxpool2(x)

        x = self.proj3(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, [idx1, idx2], [x_fea1, x_fea2]


class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7, RC_heads=1, NC_heads=6, dilations=[1, 2, 3, 4],
                RC_op='cat', RC_tokens_type='performer', NC_tokens_type='transformer', RC_group=1, NC_group=64, NC_depth=2, dpr=0.1, mlp_ratio=4., qkv_bias=True, 
                qk_scale=None, drop=0, attn_drop=0., norm_layer=nn.LayerNorm, class_token=False, gamma=False, init_values=1e-4, SE=False):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = to_2tuple(img_size // self.downsample_ratios)
        self.RC_kernel_size = kernel_size
        self.RC_heads = RC_heads
        self.NC_heads = NC_heads
        self.dilations = dilations
        self.RC_op = RC_op
        self.RC_tokens_type = RC_tokens_type
        self.RC_group = RC_group
        self.NC_group = NC_group
        self.NC_depth = NC_depth
        if downsample_ratios > 2:
            self.RC = PatchEmbed(img_size=self.img_size, embed_dim=token_dims, norm_layer=nn.LayerNorm)
        elif downsample_ratios == 2:
            self.RC = PatchMerging(input_resolution=self.img_size, in_dim=in_chans, out_dim=token_dims)
        else:
            self.RC = nn.Identity()
        
        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr, norm_layer=norm_layer, class_token=class_token, group=NC_group, tokens_type=NC_tokens_type,
                       gamma=gamma, init_values=init_values, SE=SE)
        for i in range(NC_depth)])

    def forward(self, x, input_resolution=None):
        if input_resolution is None:
            input_resolution = self.img_size
        x, indices, feas = self.RC(x, input_resolution=input_resolution)
        input_resolution = [input_resolution[0]//self.downsample_ratios, input_resolution[1]//self.downsample_ratios]
        for nc in self.NC:
            x = nc(x, input_resolution=input_resolution)
        return x, indices, feas


class ViTAE_noRC_MaxPooling_basic(nn.Module):
    def __init__(self, img_size=224, in_chans=3, stages=4, embed_dims=64, token_dims=64, downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 3, 3, 3], 
                RC_heads=[1, 1, 1, 1], NC_heads=4, dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                RC_op='cat', RC_tokens_type=['performer', 'transformer', 'transformer', 'transformer'], NC_tokens_type='transformer',
                RC_group=[1, 1, 1, 1], NC_group=[1, 32, 64, 64], NC_depth=[2, 2, 6, 2], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000,
                gamma=False, init_values=1e-4, SE=False):
        super().__init__()
        self.num_classes = num_classes
        self.stages = stages
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.RC_heads = repeatOrNot(RC_heads, stages)
        self.NC_heads = repeatOrNot(NC_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.RC_op = repeatOrNot(RC_op, stages)
        self.RC_tokens_type = repeatOrNot(RC_tokens_type, stages)
        self.NC_tokens_type = repeatOrNot(NC_tokens_type, stages)
        self.RC_group = repeatOrNot(RC_group, stages)
        self.NC_group = repeatOrNot(NC_group, stages)
        self.NC_depth = repeatOrNot(NC_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)

        self.pos_drop = nn.Dropout(p=drop_rate)
        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i==0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i], self.NC_depth[i], dpr[startDpr:self.NC_depth[i]+startDpr],
                mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i], drop=self.drop[i], attn_drop=self.attn_drop[i],
                norm_layer=self.norm_layer[i], gamma=gamma, init_values=init_values, SE=SE)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]
        self.layers = nn.ModuleList(Layers)

        # Classifier head
        self.head = nn.Linear(self.tokens_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        # load state dict here TODO

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        indices = []
        feas = []
        B, C, H, W = x.shape
        input_resolution = [H, W]

        for layer_idx in range(len(self.layers)):
            x, idx, fea = self.layers[layer_idx](x, input_resolution=input_resolution)
            indices = indices + idx
            feas = feas + fea
            input_resolution = [input_resolution[0]//self.downsample_ratios[layer_idx], input_resolution[1]//self.downsample_ratios[layer_idx]]

        # x = x.view(B, -1, input_resolution[0], input_resolution[1]).contiguous()
        x = x.view(B, input_resolution[0], input_resolution[1], -1).permute(0,3,1,2).contiguous()
        return x, indices, feas

    def forward(self, x):
        return self.forward_features(x)

    def train(self, mode=True, tag='default'):
        self.training = mode
        if tag == 'default':
            for module in self.modules():
                if module.__class__.__name__ != 'ViTAE_noRC_MaxPooling_basic':
                    module.train(mode)
        elif tag == 'linear':
            for module in self.modules():
                if module.__class__.__name__ != 'ViTAE_noRC_MaxPooling_basic':
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
        elif tag == 'linearLNBN':
            for module in self.modules():
                if module.__class__.__name__ != 'ViTAE_noRC_MaxPooling_basic':
                    if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm2d):
                        module.train(mode)
                        for param in module.parameters():
                            param.requires_grad = True
                    else:
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False
        self.head.train(mode)
        for param in self.head.parameters():
            param.requires_grad = True
        return self
