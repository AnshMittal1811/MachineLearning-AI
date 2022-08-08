import torch.nn as nn


class NoneTransform(nn.Module):
    def __init__(self, opt):
        super().__init__()

    def forward(self, x, data):
        return x
