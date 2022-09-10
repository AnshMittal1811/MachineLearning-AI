# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from lib.config import config
from lib.modeling.detector.poolers import Pooler
from lib.modeling.backbone.make_layers import make_conv3x3
from lib.modeling.detector.roi_heads.box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, in_channels):
        super().__init__()

        resolution = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        layers = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.CONV_LAYERS
        dilation = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=False
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


def make_roi_mask_feature_extractor(in_channels):
    if config.MODEL.INSTANCE2D.FPN:
        return MaskRCNNFPNFeatureExtractor(in_channels)
    else:
        return ResNet50Conv5ROIFeatureExtractor()
