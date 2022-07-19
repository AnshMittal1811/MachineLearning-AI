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
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

import torchvision
class MyViT(nn.Module):
    def __init__(self, num_classes=119035, pretrain_path=None, enable_fc=False):
        super().__init__()
        print('initializing ViT model as backbone using ckpt:', pretrain_path)
        self.model = timm.create_model('vit_base_patch16_224',checkpoint_path=pretrain_path,num_classes=num_classes)# pretrained=True)
    def forward_features(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.model.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)

        return self.model.pre_logits(x[:, 0])


    def forward(self, x):
        features = self.forward_features(x)
        return features


def timm_beit(**kwargs):

    default_kwargs={}
    default_kwargs.update(**kwargs)
    return MyViT(**default_kwargs)

def test_build():
    model = MyViT(init_ckpt='vit_base_patch32_224_in21k')
    image = torch.rand(2, 3, 224, 224)
    output = model(image)

if __name__ == '__main__':
    test_build()
