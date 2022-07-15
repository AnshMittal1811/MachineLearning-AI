import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import PanoUpsampleW


class Upsample1D(nn.Sequential):
    def __init__(self, ic, oc):
        super(Upsample1D, self).__init__(
            PanoUpsampleW(4),
            nn.Conv1d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm1d(oc),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat):
        feat1d = feat['1D']
        for module in self:
            feat1d = module(feat1d)
        feat['1D'] = feat1d
        return feat
