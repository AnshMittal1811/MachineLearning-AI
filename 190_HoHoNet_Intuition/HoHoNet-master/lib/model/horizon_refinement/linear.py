import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1dbnrelu(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, **kwargs),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )

class Linear(nn.Module):
    def __init__(self, c_mid, base_ch=256):
        super(Linear, self).__init__()
        self.conv_1x1 = conv1dbnrelu(c_mid, base_ch*4, kernel_size=1, bias=False)
        self.out_channels = base_ch*4

    def forward(self, feat):
        feat = feat['1D']
        feat = self.conv_1x1(feat)
        return {'1D': feat}
