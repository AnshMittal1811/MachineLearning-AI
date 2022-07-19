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
class Myinceptionv4(nn.Module):
    def __init__(self, num_classes=1001, pretrain_path=None, enable_fc=False):
        super().__init__()
        print('initializing inception_v4 model as backbone using ckpt:', pretrain_path)
        self.model = timm.create_model('inception_v4',checkpoint_path=pretrain_path,num_classes=num_classes)# pretrained=True)
    def forward_features(self, x):
        return self.model.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.global_pool(x)
        return x



def timm_inceptionv4(**kwargs):

    default_kwargs={}
    default_kwargs.update(**kwargs)
    return Myinceptionv4(**default_kwargs)

def test_build():
    model = Myinceptionv4(clip_pretrain_path='/mnt/lustre/zhangyuanhan/architech/inceptionv4-8e4777a0.pth')
    image = torch.rand(2, 3, 299, 299)
    output = model(image)
    print(output.shape) #768

if __name__ == '__main__':
    test_build()
