# https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import random
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from einops import rearrange
from torch import nn, Tensor

__all__ = ["LayoutGenerator"]

from models.networks.stylegan import StyleMLP, pixel_norm
from utils import derange_tensor


@dataclass(eq=False)
class LayoutGenerator(nn.Module):
    noise_dim: int = 512
    feature_dim: int = 512
    style_dim: int = 512
    # MLP options
    mlp_n_layers: int = 8
    mlp_trunk_n_layers: int = 4
    mlp_hidden_dim: int = 1024
    n_features_max: int = 5
    norm_features: bool = False
    # Transformer options
    spatial_style: bool = False
    # Training options
    mlp_lr_mul: float = 0.01
    shuffle_features: bool = False
    p_swap_style: float = 0.0
    feature_jitter_xy: float = 0.0  # Legacy, unused
    feature_dropout: float = 0.0
    shift_bias: float = 0.0
    shift_scale: float = 1.0

    def __post_init__(self):
        super().__init__()
        if self.feature_jitter_xy:
            print('Warning! This parameter is here only to support loading of old checkpoints, and does not function.'
                  'Unless you are loading a model that has this value set, it should not be used. To control jitter, '
                  'set model.feature_jitter_xy directly.')
        # {x_i, y_i, feature_i, covariance_i}, bg feature, and cluster sizes
        maybe_style_dim = int(self.spatial_style) * self.style_dim
        ndim = (self.feature_dim + maybe_style_dim + 2 + 4 + 1) * self.n_features_max + \
               (maybe_style_dim + self.feature_dim + 1)
        self.mlp = StyleMLP(self.mlp_n_layers, self.mlp_hidden_dim, self.mlp_lr_mul, first_dim=self.noise_dim,
                            last_dim=ndim, last_relu=False)

    def forward(self, noise: Tensor, n_features: int,
                mlp_idx: Optional[int] = None) -> Optional[Dict[str, Tensor]]:
        """
        Args:
            noise: [N x noise_dim] or [N x M x noise_dim]
            mlp_idx: which IDX to start running MLP from, useful for truncation
            n_features: int num features to output
        Returns: three tensors x coordinates [N x M], y coordinates [N x M], features [N x M x feature_dim]
        """
        if mlp_idx is None:
            out = self.mlp(noise)
        else:
            out = self.mlp[mlp_idx:](noise)
        sizes, out = out.tensor_split((self.n_features_max + 1,), dim=1)
        bg_feat, out = out.tensor_split((self.feature_dim,), dim=1)
        if self.spatial_style:
            bg_style_feat, out = out.tensor_split((self.style_dim,), dim=1)
        out = rearrange(out, 'n (m d) -> n m d', m=self.n_features_max)
        if self.shuffle_features:
            idxs = torch.randperm(self.n_features_max)[:n_features]
        else:
            idxs = torch.arange(n_features)
        out = out[:, idxs]
        sizes = sizes[:, [0] + idxs.add(1).tolist()]
        if self.feature_dropout:
            keep = torch.rand((out.size(1),)) > self.feature_dropout
            if not keep.any():
                keep[0] = True
            out = out[:, keep]
            sizes = sizes[:, [True] + keep.tolist()]
        xy = out[..., :2].sigmoid()  # .mul(self.max_coord)
        ret = {'xs': xy[..., 0], 'ys': xy[..., 1], 'sizes': sizes[:, :n_features + 1], 'covs': out[..., 2:6]}
        end_dim = self.feature_dim + 6
        features = out[..., 6:end_dim]
        features = torch.cat((bg_feat[:, None], features), 1)
        ret['features'] = features
        # return [xy[..., 0], xy[..., 1], features, covs, sizes[:, :n_features + 1].softmax(-1)]
        if self.spatial_style:
            style_features = out[..., end_dim:]
            style_features = torch.cat((bg_style_feat[:, None], style_features), 1)
            ret['spatial_style'] = style_features
        ret['covs'] = ret['covs'].detach()
        if self.norm_features:
            for k in ('features', 'spatial_style', 'shape_features'):
                if k in ret:
                    ret[k] = pixel_norm(ret[k])
        if self.p_swap_style:
            if random.random() <= self.p_swap_style:
                n = random.randint(0, ret['spatial_style'].size(1) - 1)
                shuffle = torch.randperm(ret['spatial_style'].size(1) - 1).add(1)[:n]
                ret['spatial_style'][:, shuffle] = derange_tensor(ret['spatial_style'][:, shuffle])
        return ret
