""" CUDA / AMP utils
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from timm.utils import *

__all__ = ['NativeScaler']

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, f, data_iter, data_loader, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        data_iter = self._scaler.step(optimizer, f, data_iter, data_loader)
        self._scaler.update()
        return data_iter

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)