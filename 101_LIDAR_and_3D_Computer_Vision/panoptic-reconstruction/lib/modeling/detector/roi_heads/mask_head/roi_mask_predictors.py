# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from lib.config import config


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels

        self.conv5_mask = nn.ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(in_channels):
    return MaskRCNNC4Predictor(in_channels)
