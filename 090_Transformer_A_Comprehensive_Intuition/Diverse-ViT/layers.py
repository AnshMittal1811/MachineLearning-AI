import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)        
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))

    def forward(self, input):
        W = self.weight_mask * self.weight
        b = self.bias
        return F.linear(input, W, b)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1,
                bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        W = self.weight_mask * self.weight
        b = self.bias
        return self._conv_forward(input, W, b)
