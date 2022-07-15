import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bases
from ..utils import PanoUpsampleW


''' Dense (per-pixel) depth estimation '''
class DepthBase(nn.Module):
    def __init__(self):
        super(DepthBase, self).__init__()

    def infer(self, x_emb):
        depth = self(x_emb)['depth']
        return {'depth': depth}

    def compute_losses(self, x_emb, batch):
        gt = batch['depth']
        mask = (gt > 0)

        # Forward
        pred_dict = self(x_emb)
        pred = pred_dict['depth']

        # Compute losses
        losses = {}
        l1 = (pred[mask] - gt[mask]).abs()
        l2 = (pred[mask] - gt[mask]).pow(2)
        losses['mae'] = l1.mean()
        losses['rmse'] = l2.mean().sqrt()
        losses['delta1'] = (torch.max(pred[mask]/gt[mask], gt[mask]/pred[mask]) < 1.25).float().mean()

        losses['total.depth'] = loss_for_backward(pred_dict['depth1d'], gt, mask, self.loss)
        if 'residual' in pred_dict:
            with torch.no_grad():
                gt_residual = gt - pred_dict['depth1d'].detach()
            losses['total.residual'] = loss_for_backward(pred_dict['residual'], gt_residual, mask, 'l1')
        return losses


def loss_for_backward(pred, gt, mask, loss):
    if loss == 'l1':
        return F.l1_loss(pred[mask], gt[mask])
    elif loss == 'l2':
        return F.mse_loss(pred[mask], gt[mask])
    elif loss == 'huber':
        return F.smooth_l1_loss(pred[mask], gt[mask])
    elif loss == 'berhu':
        l1 = (pred[mask] - gt[mask]).abs().mean()
        l2 = (pred[mask] - gt[mask]).pow(2).mean()
        with torch.no_grad():
            c = max(l1.detach().max() * 0.2, 0.01)
        l2c = (l2 + c**2) / (2 * c)
        return torch.where(l1<=c, l1, l2c).mean()
    else:
        raise NotImplementedError


class DepthEstimator(DepthBase):
    def __init__(self, emb_dim, basis='dct', loss='l1', n_components=64,
                 init_weight=0.1, init_bias=2.5, output_height=512,
                 resisual=False, basis_tuning=False):
        super(DepthEstimator, self).__init__()
        self.loss = loss

        self.output_height = output_height
        basis = getattr(bases, basis)(n_components, output_height)
        if basis_tuning:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer('basis', basis)

        self.estimator = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 1),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(emb_dim, n_components, 1, bias=False),
        )
        self.bias = nn.Parameter(torch.full([1], init_bias))
        nn.init.normal_(self.estimator[-1].weight, std=init_weight/np.sqrt(emb_dim/2))

        self.residual = None
        if resisual:
            self.residual = nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1, bias=False),
                PanoUpsampleW(4),
                nn.UpsamplingBilinear2d(scale_factor=(4,1)),
            )

    def forward(self, x_emb):
        ws = self.estimator(x_emb['1D'])
        if self.basis is None:
            h, w = self.output_height, ws.shape[-1]
            depth = self.bias + F.interpolate(ws.unsqueeze(1), size=(h,w), mode='bilinear', align_corners=False)
        else:
            depth = self.bias + torch.einsum('bkw,kh->bhw', ws, self.basis).unsqueeze(1)
        ret_dict = {'depth': depth, 'depth1d': depth}
        if self.residual is not None:
            residual = 0.1 * self.residual(x_emb['conv_list'][0].detach())
            ret_dict['residual'] = residual
            ret_dict['depth'] = depth + residual
        return ret_dict
