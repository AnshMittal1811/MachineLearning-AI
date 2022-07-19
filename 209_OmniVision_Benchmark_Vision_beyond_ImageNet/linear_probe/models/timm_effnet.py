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
class Myeffnet(nn.Module):
    def __init__(self, num_classes=1000, pretrain_path=None, enable_fc=False):
        super().__init__()
        print('initializing beit model as backbone using ckpt:', pretrain_path)
        self.model = timm.create_model('efficientnet_b4',checkpoint_path=pretrain_path,num_classes=num_classes)# pretrained=True)
    def forward_features(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.blocks(x)
        x = self.model.conv_head(x)
        x = self.model.bn2(x)
        x = self.model.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.global_pool(x)
        # if self.drop_rate > 0.:
        #     x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


def timm_effnet(**kwargs):

    default_kwargs={}
    default_kwargs.update(**kwargs)
    return Myeffnet(**default_kwargs)

def test_build():
    model = Myeffnet(clip_pretrain_path='/mnt/lustre/zhangyuanhan/architech/efficientnet_b4_ra2_320-7eb33cd5.pth')
    image = torch.rand(2, 3, 224, 224)
    output = model(image)
    print(output.shape)#1792

if __name__ == '__main__':
    test_build()
