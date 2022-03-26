#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from logging import raiseExceptions
import slowfast.utils.logging as logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

logger = logging.get_logger(__name__)

class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, reduction, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EKLoss(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean", ce_type='', smoothing=0.1):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super().__init__()
        self.reduction = reduction
        if ce_type == 'soft':
            self.ce_loss = SoftTargetCrossEntropy(reduction = self.reduction)
        elif ce_type == 'label_smoothing':
            self.ce_loss = LabelSmoothingCrossEntropy(reduction = self.reduction, smoothing=smoothing)
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction = self.reduction)

    def forward(self, extra_preds, y):
        verb = extra_preds['verb']
        noun = extra_preds['noun']
        verb_loss = self.ce_loss(verb, y['verb'])
        noun_loss = self.ce_loss(noun, y['noun'])
        return {'verb_loss':verb_loss,  'noun_loss':noun_loss}

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "label_smoothing_cross_entropy": LabelSmoothingCrossEntropy,
}

def get_loss_func(cfg, state = 'train'):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    loss_name = cfg.MODEL.LOSS_FUNC
    if state == 'val' and loss_name == 'soft_cross_entropy':
        loss_name = 'cross_entropy'
    if cfg.TRAIN.DATASET == 'epickitchens':
        if loss_name == 'cross_entropy':
            ret = partial(EKLoss, ce_type='')
        elif loss_name == 'soft_cross_entropy':
            ret = partial(EKLoss, ce_type='soft')
        elif loss_name == 'label_smoothing_cross_entropy':
            ret = partial(EKLoss, ce_type='label_smoothing', smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE)
        else:
            raise NotImplementedError(f"{loss_name} for epickitchens")
    else:
        if loss_name not in _LOSSES.keys():
            raise NotImplementedError("Loss {} is not supported".format(loss_name))
        ret = _LOSSES[loss_name]
        if loss_name == "label_smoothing_cross_entropy":
            ret = partial(ret, smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE)
    return ret
