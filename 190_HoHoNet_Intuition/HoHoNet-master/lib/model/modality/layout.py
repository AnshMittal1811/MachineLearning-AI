import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bases

from lib.misc import panostretch, post_proc
from ..utils import peaks_finding
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon


''' Layout (per-column) estimation '''
class LayoutEstimator(nn.Module):
    def __init__(self, emb_dim, bon_weight=1., cor_weight=1., bon_loss='l1', cor_loss='bce', bon_scale=1.,
                 init_weight=0.1, dropout=0., oneconv=True, last_ks=1, last_bias=True,
                 H=512, W=1024, post_force_cuboid=False):
        super(LayoutEstimator, self).__init__()
        self.bon_loss = bon_loss
        self.cor_loss = cor_loss
        self.bon_scale = bon_scale
        self.bon_weight = bon_weight
        self.cor_weight = cor_weight
        self.H = H
        self.W = W
        self.post_force_cuboid = post_force_cuboid

        if oneconv:
            self.pred_bon = nn.Conv1d(emb_dim, 2, last_ks, padding=last_ks//2, bias=last_bias)
            self.pred_cor = nn.Conv1d(emb_dim, 1, last_ks, padding=last_ks//2, bias=last_bias)
            if last_bias:
                nn.init.constant_(self.pred_bon.bias[0], -0.478)
                nn.init.constant_(self.pred_bon.bias[1], 0.425)
                nn.init.constant_(self.pred_cor.bias, -1.)
        else:
            self.pred_bon = nn.Sequential(
                nn.Conv1d(emb_dim, emb_dim, 3, padding=1, bias=False),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(emb_dim, 2, 1),
            )
            self.pred_cor = nn.Sequential(
                nn.Conv1d(emb_dim, emb_dim, 3, padding=1, bias=False),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(emb_dim, 1, 1),
            )
            nn.init.constant_(self.pred_bon[-1].bias[0], -0.478)
            nn.init.constant_(self.pred_bon[-1].bias[1], 0.425)
            nn.init.constant_(self.pred_cor[-1].bias, -1.)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x_emb):
        x_emb = x_emb['1D']
        if self.dropout is not None:
            x_emb = self.dropout(x_emb)
        pred_bon = self.pred_bon(x_emb)
        pred_cor = self.pred_cor(x_emb)
        return {'bon': pred_bon, 'cor': pred_cor}

    def infer(self, x_emb):
        pred = self(x_emb)
        pred_bon = pred['bon'] / self.bon_scale
        pred_cor = pred['cor']
        H, W = self.H, self.W

        y_bon_ = (pred_bon[0].cpu().numpy() / np.pi + 0.5) * H - 0.5
        y_cor_ = pred_cor[0,0].sigmoid().cpu().numpy()
        # Init floor/ceil plane
        z0 = 50
        _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

        # Detech wall-wall peaks
        def find_N_peaks(signal, r, min_v, N):
            max_v = maximum_filter(signal, size=r, mode='wrap')
            pk_loc = np.where(max_v == signal)[0]
            pk_loc = pk_loc[signal[pk_loc] > min_v]
            if N is not None:
                order = np.argsort(-signal[pk_loc])
                pk_loc = pk_loc[order[:N]]
                pk_loc = pk_loc[np.argsort(pk_loc)]
            return pk_loc, signal[pk_loc]
        min_v = 0 if self.post_force_cuboid else 0.05
        r = int(round(W * 0.05 / 2))
        N = 4 if self.post_force_cuboid else None
        xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

        # Generate wall-walls
        cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=self.post_force_cuboid)
        if not self.post_force_cuboid:
            # Check valid (for fear self-intersection)
            xy2d = np.zeros((len(xy_cor), 2), np.float32)
            for i in range(len(xy_cor)):
                xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
            if not Polygon(xy2d).is_valid:
                import sys
                print(
                    'Fail to generate valid general layout!! '
                    'Generate cuboid as fallback.',
                    file=sys.stderr)
                xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
                cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

        # Expand with btn coory
        cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
        # Collect corner position in equirectangular
        cor_id = np.zeros((len(cor)*2, 2), np.float32)
        for j in range(len(cor)):
            cor_id[j*2] = cor[j, 0], cor[j, 1]
            cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
        return {'cor_id': cor_id, 'y_bon_': y_bon_, 'y_cor_': y_cor_}

    def compute_losses(self, x_emb, batch):
        gt_bon = batch['bon'] * self.bon_scale
        gt_vot = batch['vot']
        gt_cor = 0.96 ** gt_vot.abs()

        # Forward
        pred = self(x_emb)

        # Compute losses
        losses = {}
        if self.bon_loss == 'l1':
            losses['bon'] = F.l1_loss(pred['bon'], gt_bon)
        elif self.bon_loss == 'l2':
            losses['bon'] = F.mse_loss(pred['bon'], gt_bon)
        else:
            raise NotImplementedError

        if self.cor_loss == 'bce':
            losses['cor'] = F.binary_cross_entropy_with_logits(pred['cor'], gt_cor)
        elif self.cor_loss == 'prfocal':
            g, p = gt_cor, pred['cor']
            pos_mask = (g >= 1-1e-6)
            B, alpha, beta = len(g), 2, 4
            L_pos = -F.logsigmoid(p) * F.sigmoid(-p).pow(alpha)
            L_neg = -F.logsigmoid(-p) * F.sigmoid(p).pow(alpha) * (1-g).pow(beta)
            L = torch.where(pos_mask, L_pos, L_neg).view(B,-1).sum(-1) / pos_mask.float().view(B,-1).sum(-1)
            losses['cor'] = L.mean()
        else:
            raise NotImplementedError

        losses['total.layout'] = self.bon_weight * losses['bon'] + self.cor_weight * losses['cor']
        with torch.no_grad():
            losses['bon.mae'] = F.l1_loss(pred['bon'], gt_bon) / self.bon_scale
            losses['cor.mae'] = F.l1_loss(pred['cor'].sigmoid(), gt_cor)
        return losses
