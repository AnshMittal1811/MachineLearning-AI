import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=dilation, stride=stride, dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



import torch.utils.checkpoint as checkpoint

DIM=32

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, HR=False):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.HR = HR

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM,  stride=1, dilation=1)
        self.layer2 = self._make_layer(2*DIM, stride=2, dilation=1)
        if not HR:
            self.layer3 = self._make_layer(4*DIM, stride=2)

        self.conv2 = nn.Conv2d(4*DIM if not HR else 2 * DIM, output_dim, kernel_size=1)


        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def _flatten_batch_dims(self, x):
        batch_dims = x.shape[:-3]
        batch_prod = torch.Size(torch.prod(torch.tensor(batch_dims))[None])
        
        x_flat = x.reshape(batch_prod + x.shape[-3:])
        unflatten_fn = lambda y: y.reshape(batch_dims + y.shape[-3:])
        return x_flat, unflatten_fn

    def _base(self):
        def fn(x):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
            return x
        
        return fn

    def forward(self, x):

        # force 4d tensor
        x, unflatten_fn = self._flatten_batch_dims(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        if not self.HR:
            x = self.layer3(x)
        
        x = self.conv2(x)

        x = unflatten_fn(x)
        return x


class SameResEncoder(nn.Module):
    def __init__(self, output_dim=128, input_dim=3, norm_fn='batch', dropout=0.0):
        super(SameResEncoder, self).__init__()
        self.norm_fn = norm_fn
        # self.HR = HR

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, DIM, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM, stride=1, dilation=1)
        self.layer2 = self._make_layer(DIM, stride=1, dilation=1)
        self.conv2 = nn.Conv2d(DIM, output_dim, kernel_size=1)
        # if not HR:
        #     self.layer3 = self._make_layer(4 * DIM, stride=2)

        # self.conv2 = nn.Conv2d(4 * DIM if not HR else 2 * DIM, output_dim, kernel_size=1)


        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def _flatten_batch_dims(self, x):
        batch_dims = x.shape[:-3]
        batch_prod = torch.Size(torch.prod(torch.tensor(batch_dims))[None])

        x_flat = x.reshape(batch_prod + x.shape[-3:])
        unflatten_fn = lambda y: y.reshape(batch_dims + y.shape[-3:])
        return x_flat, unflatten_fn

    def _base(self):
        def fn(x):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
            return x

        return fn

    def forward(self, x):

        # force 4d tensor
        x, unflatten_fn = self._flatten_batch_dims(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        # if not self.HR:
        #     x = self.layer3(x)

        x = self.conv2(x)

        x = unflatten_fn(x)
        return x


