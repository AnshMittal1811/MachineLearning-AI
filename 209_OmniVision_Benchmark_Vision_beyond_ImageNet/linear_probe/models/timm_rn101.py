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
class Myrn101(nn.Module):
    def __init__(self, num_classes=1000, pretrain_path=None, enable_fc=False):
        super().__init__()
        print('initializing resnet101 model as backbone using ckpt:', pretrain_path)
        self.model = timm.create_model('resnet101',checkpoint_path=pretrain_path,num_classes=num_classes)# pretrained=True)

    def forward_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.global_pool(x)
        return x



def timm_rn101(**kwargs):

    default_kwargs={}
    default_kwargs.update(**kwargs)
    return Myrn101(**default_kwargs)

def test_build():
    model = Mybeit(pretrain_path='/mnt/lustre/zhangyuanhan/architech/resnet101_a1h-36d3f2aa.pth')
    image = torch.rand(2, 3, 224, 224)
    output = model(image)
    print(output.shape) #768

if __name__ == '__main__':
    test_build()
