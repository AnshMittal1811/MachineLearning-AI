import torch.nn as nn
import torch
from torchvision import models

from misc.layer import Conv2d, FC

import torch.nn.functional as F
from misc.utils import *
import torchvision.ops as ops


# model_path = '../PyTorch_Pretrained/resnet50-19c8e357.pth'

class Res50(nn.Module):
    def __init__(self,  extra_loss):
        super(Res50, self).__init__()

        self.extra_loss = extra_loss

        self.de_1 = Conv2d(1024, 256, 3, same_padding=True, NL='relu')
        self.i =1 # 0 for baseline, 1 for deform or deform loss
        if self.i==1:
            setattr(self, 'offset_w_{:d}'.format(self.i), nn.Conv2d(in_channels=256, out_channels=2*3*3, kernel_size=3, padding=1))

            # In pytorch1.8, the weight of deformable has been included into ops.DeformConv2d(So, it is good..)
            setattr(self, 'deform_{:d}'.format(self.i), ops.DeformConv2d(in_channels=256, out_channels=128, kernel_size=3, padding=2, dilation=2))
            setattr(self, 'bn_{:d}'.format(self.i), nn.BatchNorm2d(128))
        elif self.i == 0:
            self.de_2 = Conv2d(256, 128, 3, same_padding=True, NL='relu', dilation=2)
        else:
            print('set self.i = 0, 1 or 2')
            assert 1==2
            

        self.de_pred = nn.Sequential(Conv2d(128, 1, 1, same_padding=True, NL='relu'))

        initialize_weights(self.modules())

        res = models.resnet50(pretrained=True)

        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())




    def forward(self, x, out_feat=False):


        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        x = self.de_1(x)
        j=1
        x_offset_list = []
        if self.i==1:
            cur_offset = getattr(self, 'offset_w_{:d}'.format(j))
            cur_deform = getattr(self, 'deform_{:d}'.format(j))
            cur_bn = getattr(self, 'bn_{:d}'.format(j))
            x_offset = cur_offset(x)
            x = F.relu_(cur_bn(cur_deform(x, x_offset)))
        else:

            x=self.de_2(x)
        x = self.de_pred(x)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        if (self.extra_loss) and out_feat == True:
            return x, x_offset_list
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)


def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
