# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License-NC
# See LICENSE.txt for details
#
# Author: Zheng Tang (tangzhengthomas@gmail.com)


from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['DenseNet121']


class DenseNet121(nn.Module):
    def __init__(self, num_vids, num_vcolors=10, num_vtypes=9, keyptaware=True, 
                 heatmapaware=True, segmentaware=True, multitask=True, **kwargs):
        super(DenseNet121, self).__init__()
        self.keyptaware = keyptaware
        self.multitask = multitask
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features

        num_channels = 3
        if heatmapaware:
            num_channels += 36
        if segmentaware: 
            num_channels += 13

        if num_channels > 3:
            pretrained_weights = densenet121.features[0].weight
            self.base[0] = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # for other channels weights should randomly initialized with Gaussian
            self.base[0].weight.data.normal_(0, 0.02)
            self.base[0].weight.data[:, :3, :, :] = pretrained_weights

        if self.keyptaware and self.multitask:
            self.fc_vid = nn.Linear(1024 + 108, 1024)
            self.fc_vcolor = nn.Linear(1024 + 108, 512)
            self.fc_vtype = nn.Linear(1024 + 108, 512)
            self.classifier_vid = nn.Linear(1024, num_vids)
            self.classifier_vcolor = nn.Linear(512, num_vcolors)
            self.classifier_vtype = nn.Linear(512, num_vtypes)
        elif self.keyptaware:
            self.fc = nn.Linear(1024 + 108, 1024)
            self.classifier_vid = nn.Linear(1024, num_vids)
        elif self.multitask:
            self.fc_vid = nn.Linear(1024, 1024)
            self.fc_vcolor = nn.Linear(1024, 512)
            self.fc_vtype = nn.Linear(1024, 512)
            self.classifier_vid = nn.Linear(1024, num_vids)
            self.classifier_vcolor = nn.Linear(512, num_vcolors)
            self.classifier_vtype = nn.Linear(512, num_vtypes)
        else:
            self.classifier_vid = nn.Linear(1024, num_vids)
        self.feat_dim = 1024

    def forward(self, x, p=None):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if self.keyptaware and self.multitask:
            f = torch.cat((f, p.float()), dim=1)
            f_vid = F.leaky_relu(self.fc_vid(f))
            f_vcolor = F.leaky_relu(self.fc_vcolor(f))
            f_vtype = F.leaky_relu(self.fc_vtype(f))
            y_id = self.classifier_vid(f_vid)
            y_color = self.classifier_vcolor(f_vcolor)
            y_type = self.classifier_vtype(f_vtype)
            return y_id, y_color, y_type, f_vid
        elif self.keyptaware:
            f = torch.cat((f, p.float()), dim=1)
            f = F.leaky_relu(self.fc(f))
            y_id = self.classifier_vid(f)
            return y_id, f
        elif self.multitask:
            f_vid = F.leaky_relu(self.fc_vid(f))
            f_vcolor = F.leaky_relu(self.fc_vcolor(f))
            f_vtype = F.leaky_relu(self.fc_vtype(f))
            y_id = self.classifier_vid(f_vid)
            y_color = self.classifier_vcolor(f_vcolor)
            y_type = self.classifier_vtype(f_vtype)
            return y_id, y_color, y_type, f_vid
        else:
            y_id = self.classifier_vid(f)
            return y_id, f
