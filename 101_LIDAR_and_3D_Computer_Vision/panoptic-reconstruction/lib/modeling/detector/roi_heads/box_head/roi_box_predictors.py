# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from lib.config import config


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_logit, bbox_pred


def make_roi_box_predictor(in_channels):
    return FastRCNNPredictor(in_channels)
