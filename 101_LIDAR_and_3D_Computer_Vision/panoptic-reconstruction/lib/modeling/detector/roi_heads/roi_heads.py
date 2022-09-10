# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torch import nn

from lib.config import config

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head


class CombinedROIHeads(nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, heads):
        super().__init__(heads)
        if config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.USE and config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)

        if config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.USE:

            mask_features = features
            if self.training and config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x

            detections, loss_mask = self.mask(mask_features, detections, targets)

            losses.update(loss_mask)

        return detections, losses

    def inference(self, features, proposals):
        x, detections, _ = self.box(features, proposals)

        if config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.USE:

            mask_features = features
            if self.training and config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x

            detections, _ = self.mask(mask_features, detections, None)

        return detections


def build_roi_heads(in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = list()
    roi_heads.append(("box", build_roi_box_head(in_channels)))

    if config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.USE:
        roi_heads.append(("mask", build_roi_mask_head(in_channels)))

    # combine individual heads in a single module
    roi_heads = CombinedROIHeads(roi_heads)

    return roi_heads
