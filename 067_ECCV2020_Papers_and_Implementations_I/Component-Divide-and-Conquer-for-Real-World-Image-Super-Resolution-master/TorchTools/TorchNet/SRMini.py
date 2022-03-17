import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

import math

from .modules import residualBlock, upsampleBlock


class SRSubPixOneLayer(nn.Module):
    def __init__(self):
        super(SRSubPixOneLayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pix = nn.PixelShuffle(4)

    def forward(self, input):
        return F.tanh(self.pix(self.conv2(self.relu(self.conv1(input)))))


