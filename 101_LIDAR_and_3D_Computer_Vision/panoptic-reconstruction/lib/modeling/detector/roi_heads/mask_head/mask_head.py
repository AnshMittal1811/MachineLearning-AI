# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from datetime import datetime

import torch
from torch import nn

from lib.structures.bounding_box import BoxList
from lib.config import config
from lib.modeling.utils import ModuleResult

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("label")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("label")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.feature_extractor = make_roi_mask_feature_extractor(in_channels)
        self.predictor = make_roi_mask_predictor(self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor()
        self.loss_evaluator = make_roi_mask_loss_evaluator()

    def forward(self, features, proposals, targets=None) -> ModuleResult:
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # during training, only focus on positive boxes
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        if self.training and config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)

        mask_logits = self.predictor(x)
        result = self.post_processor(mask_logits, proposals)

        if not self.training:
            return result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return result, dict(loss_mask=loss_mask)


def build_roi_mask_head(in_channels):
    return ROIMaskHead(in_channels)
