# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from lib.config import config
from lib.modeling.backbone import resnet_fb as resnet
from lib.modeling.detector.poolers import Pooler
from lib.modeling.backbone.make_layers import group_norm, make_fc


class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        resolution = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        if config.MODEL.BACKBONE.CONV_BODY == "R-50":
            stage = resnet.StageSpec(index=4, block_count=6, return_features=False)
        elif config.MODEL.BACKBONE.CONV_BODY == "R-18":
            stage = resnet.StageSpec(index=2, block_count=3, return_features=False)

        if config.MODEL.FIXNORM:
            block_module = "BottleneckWithFixedBatchNorm"
        else:
            block_module = "Bottleneck"

        head = resnet.ResNetHead(
            block_module=block_module,
            stages=(stage,),
            num_groups=1,
            width_per_group=64,
            stride_in_1x1=True,
            stride_init=None,
            res2_out_channels=256,
            dilation=1
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)

        return x


def make_roi_box_feature_extractor():
    return ResNet50Conv5ROIFeatureExtractor()
