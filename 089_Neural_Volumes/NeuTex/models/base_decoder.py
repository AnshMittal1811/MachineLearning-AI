import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecoder(nn.Module):
    def forward(self, pts, viewdirs=None, encoding=None, *args, **kwargs):
        raise NotImplementedError()

    def get_loss(self):
        return dict()
