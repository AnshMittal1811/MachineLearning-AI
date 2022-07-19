# ------------------------------------------------------------------------
# SenseTime VTAB
# Copyright (c) 2021 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
VTAB-SenseTime Model & Criterion Classes
"""
import timm

import torch
import copy
import torch.nn as nn

import torchvision
class Mybeit(nn.Module):
    def __init__(self, num_classes=1000, pretrain_path=None, enable_fc=False):
        super().__init__()
        print('initializing beit model as backbone using ckpt:', pretrain_path)
        self.model = timm.create_model('beit_base_patch16_224',checkpoint_path=pretrain_path,num_classes=num_classes)# pretrained=True)
    def forward_features(self, x):
        x = self.model.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.model.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.model.pos_embed is not None:
            x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        rel_pos_bias = self.model.rel_pos_bias() if self.model.rel_pos_bias is not None else None
        for blk in self.model.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.model.norm(x)
        if self.model.fc_norm is not None:
            t = x[:, 1:, :]
            return self.model.fc_norm(t.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


def timm_beit(**kwargs):

    default_kwargs={}
    default_kwargs.update(**kwargs)
    return Mybeit(**default_kwargs)

def test_build():
    model = Mybeit(pretrain_path='/mnt/lustre/zhangyuanhan/architech/beit_base_patch16_224_pt22k_ft22kto1k.pth')
    image = torch.rand(2, 3, 224, 224)
    output = model(image)
    print(output.shape) #768

if __name__ == '__main__':
    test_build()
