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
class Mymlp_mixer(nn.Module):
    def __init__(self, num_classes=1000, pretrain_path=None, enable_fc=False):
        super().__init__()
        print('initializing mlp_mixer model as backbone using ckpt:', pretrain_path)
        self.model = timm.create_model('mixer_b16_224',checkpoint_path=pretrain_path,num_classes=num_classes)# pretrained=True)
    def forward_features(self, x):
        x = self.model.stem(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x



def timm_mlp_mixer(**kwargs):

    default_kwargs={}
    default_kwargs.update(**kwargs)
    return Mymlp_mixer(**default_kwargs)

def test_build():
    model = Mymlp_mixer(clip_pretrain_path='/mnt/lustre/zhangyuanhan/architech/jx_mixer_b16_224-76587d61.pth')
    image = torch.rand(2, 3, 224, 224)
    output = model(image)
    print(output.shape) #768

if __name__ == '__main__':
    test_build()
