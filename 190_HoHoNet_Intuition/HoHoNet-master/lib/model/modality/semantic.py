import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bases


''' Dense (per-pixel) semantic segmentation '''
class SemanticSegmenter(nn.Module):
    def __init__(self, emb_dim, num_classes, basis='dct', loss='bce', label_weight='', invalid_ids=[], n_components=64,
                 last_ks=1, dropout=0, init_weight=0.1, init_bias=None, output_height=512, pre1d=False):
        super(SemanticSegmenter, self).__init__()
        self.num_classes = num_classes
        self.loss = loss
        self.n_components = n_components
        self.invalid_ids = invalid_ids
        if init_bias is None:
            if self.loss == 'bce':
                init_bias = -np.log(num_classes-1)
            else:
                init_bias = 0.0

        self.output_height = output_height
        self.register_buffer('basis', getattr(bases, basis)(n_components, output_height))

        self.estimator = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, last_ks, padding=last_ks//2),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(emb_dim, n_components * num_classes, 1, bias=False),
        )
        if dropout > 0:
            self.estimator = nn.Sequential(*self.estimator[:-1], nn.Dropout(dropout), self.estimator[-1])
        self.bias = nn.Parameter(torch.full([1, num_classes, 1, 1], init_bias))
        nn.init.normal_(self.estimator[-1].weight, std=init_weight/np.sqrt(emb_dim/2))

        self.estimator1d = None
        if pre1d:
            self.estimator1d = nn.Sequential(
                nn.Conv1d(emb_dim, emb_dim, last_ks, padding=last_ks//2),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(emb_dim, num_classes, 1),
            )
            nn.init.constant_(self.estimator1d[-1].bias, -np.log(10-1))

        if label_weight:
            self.register_buffer('label_weight', torch.load(label_weight).float())
        else:
            self.register_buffer('label_weight', torch.ones(num_classes))
        self.label_weight[self.invalid_ids] = 0
        self.label_weight *= (num_classes - len(self.invalid_ids)) / self.label_weight.sum()

    def forward(self, x_emb):
        x_emb = x_emb['1D']
        B, _, W = x_emb.shape
        ws = self.estimator(x_emb).view(B, self.num_classes, self.n_components, W)
        if self.basis is None:
            h, w = self.output_height, ws.shape[-1]
            sem = self.bias + F.interpolate(ws, size=(h,w), mode='bilinear', align_corners=False)
        else:
            sem = self.bias + torch.einsum('bckw,kh->bchw', ws, self.basis)
        sem[:, self.invalid_ids] = -100

        if self.estimator1d is not None:
            sem1d = self.estimator1d(x_emb).view(B, self.num_classes, 1, W)
            sem1d[:, self.invalid_ids] = -100
            sem.permute(0,1,3,2)[sem1d.sigmoid().squeeze(2) < 0.1] = float("-Inf")
            return {'sem': sem, 'sem1d': sem1d}
        else:
            return {'sem': sem}

    def infer(self, x_emb):
        return self(x_emb)

    def compute_losses(self, x_emb, batch):
        gt = batch['sem']
        mask = (gt >= 0)
        B, H, W = gt.shape
        if mask.sum() == 0:
            return {}

        # Forward
        pred = self(x_emb)
        pred_sem = pred['sem']

        # Compute losses
        losses = {}

        if 'sem1d' in pred:
            pred_sem1d = pred['sem1d']
            gt1d = torch.zeros_like(pred_sem1d)
            brcid = torch.stack(torch.meshgrid(torch.arange(gt.shape[0]), torch.arange(gt.shape[1]), torch.arange(gt.shape[2])), -1)
            bid, rid, cid = brcid[mask].T
            gt1d[bid, gt[mask], 0, cid] = 1
            losses['acc.sem1d.fn'] = ((pred_sem1d.sigmoid() < 0.1) & (gt1d == 1)).float().mean()
            losses['acc.sem1d.tn'] = ((pred_sem1d.sigmoid() < 0.1) & (gt1d == 0)).float().mean()
            losses['total.sem1d'] = F.binary_cross_entropy_with_logits(pred_sem1d, gt1d)

        pred_sem = pred_sem.permute(0,2,3,1)[mask]
        gt = gt[mask]
        if 'sem1d' in pred:
            activate = (pred_sem1d.detach().sigmoid() >= 0.1).float().repeat(1,1,H,1)
            activate = activate.permute(0,2,3,1)[mask]
        else:
            activate = torch.ones_like(pred_sem)
        losses['acc'] = (pred_sem.argmax(1) == gt).float().mean()
        if self.loss == 'bce':
            gt_onehot = torch.zeros_like(pred_sem).scatter_(dim=1, index=gt[:,None], src=torch.ones_like(pred_sem))
            bce = F.binary_cross_entropy_with_logits(pred_sem, gt_onehot, reduction='none')
            bce = (bce * self.label_weight)[activate.bool()]
            losses['total.sem'] = bce.mean()
        elif self.loss == 'ce':
            ce = F.cross_entropy(pred_sem, gt, weight=self.label_weight, reduction='none')
            ce = ce[~torch.isinf(ce) & ~torch.isnan(ce)]
            losses['total.sem'] = ce.mean()
        elif self.loss.startswith('mse'):
            R = float(self.loss[3:])
            gt_R = torch.full_like(pred_sem, -R).scatter_(dim=1, index=gt[:,None], src=torch.full_like(pred_sem, R))
            mse = (pred_sem - gt_R).pow(2)
            losses['total.sem'] = (mse * self.label_weight).mean()
        else:
            raise NotImplementedError
        return losses
