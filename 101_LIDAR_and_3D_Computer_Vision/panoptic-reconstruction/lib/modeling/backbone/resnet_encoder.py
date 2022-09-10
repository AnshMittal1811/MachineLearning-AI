from typing import Dict

import torch
from torch import nn

from lib.modeling.utils import ModuleResult


class ResNetEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_channels = [self.layer1[-1].out_channels,
                             self.layer2[-1].out_channels,
                             self.layer3[-1].out_channels,
                             self.layer4[-1].out_channels]

    def forward(self, x: torch.Tensor) -> ModuleResult:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return {}, {"blocks": [x_block1, x_block2, x_block3, x_block4]}

    def test(self, x: torch.Tensor) -> Dict:
        _, result = self.forward(x)
        return result
