# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from lib.modeling.detector.box_coder import BoxCoder
from lib.modeling.detector.rpn.loss import make_rpn_loss_evaluator
from lib.modeling.detector.rpn.anchor_generator import make_anchor_generator
from lib.modeling.detector.rpn.inference import make_rpn_postprocessor


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNModule(nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, in_channels):
        super().__init__()

        anchor_generator = make_anchor_generator()

        head = RPNHead(in_channels, anchor_generator.num_anchors_per_location()[0])

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, features, targets=None):
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator([(320, 240) for _ in range(objectness[0].shape[0])], features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        # For end-to-end models, anchors must be transformed into boxes and
        # sampled into a training batch.
        with torch.no_grad():
            boxes = self.box_selector_train(anchors, objectness, rpn_box_regression, targets)

        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(anchors, objectness, rpn_box_regression, targets)
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)

        return boxes, {}

    def inference(self, features):
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator([(320, 240) for _ in range(objectness[0].shape[0])], features)
        return self._forward_test(anchors, objectness, rpn_box_regression)


def build_rpn(in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(in_channels)
