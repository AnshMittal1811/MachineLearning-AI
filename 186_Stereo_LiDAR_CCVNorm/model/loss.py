"""
Implementation of custom loss functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_no_grad(tensor):
    """ To make sure tensor of ground truth cannot backprop """
    assert not tensor.requires_grad, \
               'nn criterions don\'t compute the gradient w.r.t. targets - please ' \
               'mark these tensors as not requiring gradients'


class L1Loss(nn.Module):
    """ L1 loss but mask out invalid points (value <= 0). """
    def __init__(self, inputs_idx=0, targets_idx=0):
        super(L1Loss, self).__init__()
        self.inputs_idx = inputs_idx
        self.targets_idx = targets_idx

    def forward(self, inputs, targets):
        inputs = inputs[self.inputs_idx]
        targets = targets[self.targets_idx]
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach()
        diff = (targets - inputs)[valid_mask]
        if diff.shape[0] == 0: # make sure the case that depth are all masked out
            loss = (targets - inputs).mean() * 0
        else:
            loss = diff.abs().mean()

        return loss


class L2Loss(nn.Module):
    """ L2 loss but mask out invalid points (value <= 0). """
    def __init__(self, inputs_idx=0, targets_idx=0):
        super(L2Loss, self).__init__()
        self.inputs_idx = inputs_idx
        self.targets_idx = targets_idx

    def forward(self, inputs, targets):
        inputs = inputs[self.inputs_idx]
        targets = targets[self.targets_idx]
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach()
        diff = (targets - inputs)[valid_mask]
        if diff.shape[0] == 0: # make sure the case that depth are all masked out
            loss = (targets - inputs).mean() * 0
        else:
            loss = diff.pow(2).mean()

        return loss
    

InvDispL1Loss = L1Loss
