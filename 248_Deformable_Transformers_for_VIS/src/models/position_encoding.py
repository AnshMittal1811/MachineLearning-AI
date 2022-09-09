"""
Various positional encodings for the transformer.
Modified from VisTr (https://github.com/Epiphqny/VisTR) & Deformable DETR https://github.com/fundamentalvision/Deformable-DETR
"""
import math
import torch
from torch import nn
from torch.nn.init import normal_
from ..util.misc import NestedTensor


class PositionEmbeddingSpatialTemporalSine(nn.Module):
    """
    Adapted from VisTr, positional encoding for x,y,t  simultaneously. Pads with 0's if hidden size is not divisible by 3.
    """

    def __init__(self, num_pos_feats=64, num_frames=36, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.frames = num_frames
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        n, h, w = mask.shape
        mask = mask.reshape(n // self.frames, self.frames, h, w)
        assert mask is not None
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)

        pad = torch.zeros_like(pos)[:, :, :4]
        pos = torch.cat([pos, pad], dim=2)

        return pos


class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D sine positional encoding
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((
            pos_x[:, :, :, 0::2].sin(),
            pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((
            pos_y[:, :, :, 0::2].sin(),
            pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class PositionEmbeddingSineWithLearnableTemporal(PositionEmbeddingSine):
    """
    Extends 2D x,y sine positional encoding with learned temporal embedding
    """

    def __init__(self, hidden_dim=256, num_frames=6, temperature=10000, normalize=False, scale=None):
        super().__init__(num_pos_feats=hidden_dim // 2, temperature=temperature, normalize=normalize, scale=scale)
        self.num_frames = num_frames
        self.temporal_embed = nn.Parameter(torch.Tensor(num_frames, hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        normal_(self.temporal_embed)

    def forward(self, tensor_list: NestedTensor):
        pos_xy = super().forward(tensor_list)
        pos_z = self.temporal_embed[:, :, None, None]
        pos = pos_xy + pos_z
        return pos


def build_position_encoding(cfg):
    if cfg.DATASETS.TYPE == "vis":
        if cfg.MODEL.DEVIS.TEMPORAL_EMBEDDING == 'learned':
            position_embedding = PositionEmbeddingSineWithLearnableTemporal(cfg.MODEL.HIDDEN_DIM, num_frames=cfg.MODEL.DEVIS.NUM_FRAMES, normalize=True)

        elif cfg.MODEL.DEVIS.TEMPORAL_EMBEDDING == 'sine':
            # TODO: Implement hidden size != 252
            assert cfg.MODEL.HIDDEN_DIM == 252
            position_embedding = PositionEmbeddingSpatialTemporalSine(84, num_frames=cfg.MODEL.DEVIS.NUM_FRAMES, normalize=True)

        else:
            raise NotImplementedError(f"Selected DeVIS temporal positional encoding: {cfg.MODEL.DEVIS.TEMPORAL_EMBEDDING} not available. Options: [sine, learned]")

    else:
        position_embedding = PositionEmbeddingSine(cfg.MODEL.HIDDEN_DIM // 2, normalize=True)

    return position_embedding
