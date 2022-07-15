import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, c_mid, *args, **kwargs):
        super(Identity, self).__init__()
        self.out_channels = c_mid

    def forward(self, x):
        return x
