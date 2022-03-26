import torch
from torch import nn
import torch.nn.functional as F

from .utils import ObjectsCrops, box2spatial_layout, Mlp

from slowfast.models.attention import TrajectoryAttention, SeltAttentionBlock
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ORViT(nn.Module):

    def __init__(
            self, cfg, dim=768, dim_out=None, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code = False, nb_frames=None,
        ):
        super().__init__()

        self.cfg = cfg
        self.in_dim = dim
        self.dim = dim
        self.nb_frames = nb_frames

        self.with_cls_token = True 
        self.with_motion_stream = cfg.ORVIT.USE_MOTION_STREAM

        # Object Tokens
        self.crop_layer = ObjectsCrops(cfg)
        self.patch_to_d = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // 2, self.dim, bias=False),
            nn.ReLU()
        )

        self.box_categories = nn.Parameter(torch.zeros(self.nb_frames, self.cfg.ORVIT.O, self.in_dim))
        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.in_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim // 2, self.in_dim, bias=False),
            nn.ReLU()
        )

        # Attention Block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = TrajectoryAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop,
            proj_drop=drop,
        )


        if self.with_motion_stream:
            self.motion_stream = MotionStream(cfg, dim=dim, num_heads=num_heads, attn_type=cfg.ORVIT.MOTION_STREAM_ATTN_TYPE, 
                                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, 
                                                drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
                                                nb_frames=self.nb_frames,
                                                )
            self.motion_mlp = Mlp(in_features=cfg.ORVIT.MOTION_STREAM_DIM if cfg.ORVIT.MOTION_STREAM_DIM > 0 else dim,
                                        hidden_features=mlp_hidden_dim, out_features=dim,
                                        act_layer=act_layer, drop=drop)

        if self.cfg.ORVIT.INIT_WEIGHTS:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        else:
           for p in m.parameters(): nn.init.normal_(p, std=0.02)

    def forward(self,x, metadata,thw):

        box_tensors = metadata['orvit_bboxes']
        assert box_tensors is not None

        if self.with_cls_token:
            cls_token, patch_tokens = x[:,[0]], x[:,1:]

        BS, _, d = x.shape
        T,H,W = thw
        patch_tokens = patch_tokens.permute(0,2,1).reshape(BS, -1, T,H,W)
        
        BS, d, T, H, W = patch_tokens.shape
        assert T == self.nb_frames

        Tratio = box_tensors.shape[1] // T
        box_tensors = box_tensors[:,::Tratio] # [BS, T , O, 4]
        O = box_tensors.shape[-2]

        object_tokens = self.crop_layer(patch_tokens, box_tensors)  # [BS, O,T, d, H, W]
        object_tokens = object_tokens.permute(0, 1,2,4,5,3)  # [BS, O,T, H, W, d]
        object_tokens = self.patch_to_d(object_tokens) # [BS,O,T, H, W, d]
        object_tokens =torch.amax(object_tokens, dim=(-3,-2)) # [BS, O,T, d]
        object_tokens = object_tokens.permute(0,2,1,3)

        box_categories = self.box_categories.unsqueeze(0).expand(BS,-1,-1,-1)
        box_emb = self.c_coord_to_feature(box_tensors)
        object_tokens = object_tokens + box_categories + box_emb # [BS, T, O, d]

        all_tokens = torch.cat([patch_tokens.permute(0,2,3,4,1).reshape(BS, T, H*W, d), object_tokens], dim = 2).flatten(1,2) # [BS, T * (H*W+O),d]
        if self.with_cls_token:
            all_tokens =  torch.cat([cls_token, all_tokens], dim = 1) # [BS, 1 + T*N, d]

        all_tokens, thw = self.attn(
                    self.norm1(all_tokens), 
                    [T, H*W + O, 1], 
                )

        if self.with_cls_token:
            cls_token, all_tokens =  all_tokens[:, [0]], all_tokens[:, 1:]

        patch_tokens = all_tokens.reshape(BS,T,H*W+O,d)[:,:,:H*W].reshape(BS,T*H*W,d)


        if self.with_motion_stream:
            motion_emb = self.motion_stream(box_tensors,H, W) # [BS, T, H, W, d]
            motion_emb = self.motion_mlp(motion_emb) # [BS, T*H*W, d]
            patch_tokens = patch_tokens + motion_emb


        if self.with_cls_token:
            patch_tokens = torch.cat([cls_token, patch_tokens], dim = 1) # [BS, 1 + N, d]

        x = x + self.drop_path(patch_tokens) # [BS, N, d]
        x = x + self.drop_path(self.mlp(self.norm2(x))) # [BS, N, d]

        return x, thw

class Object2Spatial(nn.Module):
    def __init__(self, cfg, _type):
        super().__init__()
        self.cfg = cfg
        self._type = _type 
    def forward(self, all_features, context, boxes, H, W, t_avg_pooling = False):
        BS, T, O, d = all_features.shape

        if self._type == 'layout':
            ret = box2spatial_layout(boxes, all_features,H,W) # [B, d, T, H, W]
            ret = ret.permute(0,2,3,4,1)
            if t_avg_pooling:
                BS, T, H, W, d = ret.size()
                Tratio = int(T / self.cfg.MF.TEMPORAL_RESOLUTION)
                if Tratio > 1:
                    ret = ret.reshape(BS, -1, Tratio, H, W, d).mean(2)
            ret = ret.flatten(1,3) # [BS, T*H*W, d]
        elif self._type == 'spatial_only':
            assert context is not None
            ret = context.flatten(1,-2) # [BS, T*H*W, d]
        elif self._type == 'object_pooling':
            ret = torch.amax(all_features, dim = 2) # [BS, T, d]
            ret = ret.reshape(BS, T, 1,1, d).expand(BS, T, H, W, d).flatten(1,3)
        elif self._type == 'all_pooling':
            ret = torch.amax(all_features, dim = [1,2]) # [BS, T, d]
            ret = ret.reshape(BS, 1, 1,1, d).expand(BS, T, H, W, d).flatten(1,3)
        else:
            raise NotImplementedError(f'{self._type}')
        return ret

class MotionStream(nn.Module):
    def __init__(self, cfg, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            nb_frames = None,
        ):
        super().__init__()


        self.cfg = cfg

        self.in_dim = cfg.ORVIT.MOTION_STREAM_DIM if cfg.ORVIT.MOTION_STREAM_DIM > 0 else dim
        self.dim = dim
        self.nb_frames = nb_frames

        if self.cfg.ORVIT.MOTION_STREAM_SEP_POS_EMB:
            self.box_categories_T = nn.Parameter(torch.zeros(self.nb_frames, 1, self.in_dim))
            self.box_categories_O = nn.Parameter(torch.zeros(1, self.cfg.ORVIT.O, self.in_dim))
        else:
            self.box_categories = nn.Parameter(torch.zeros(self.nb_frames, self.cfg.ORVIT.O, self.in_dim))


        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.in_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim // 2, self.in_dim, bias=False),
            nn.ReLU()
        )

        # Attention Block
        self.attn_type = attn_type

        if attn_type == 'joint':
            self.attn = SeltAttentionBlock(
                dim=self.in_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

        self.obj2spatial = Object2Spatial(cfg, _type = 'layout')

    def forward(self, box_tensors, H, W):
        # box_tenors: [BS, T, O, 4]
        BS = box_tensors.shape[0]

        box_emb = self.c_coord_to_feature(box_tensors)

        if self.cfg.ORVIT.MOTION_STREAM_SEP_POS_EMB:
            shape = (self.nb_frames, self.cfg.ORVIT.O, self.in_dim)
            box_categories = self.box_categories_T.expand(shape) + self.box_categories_O.expand(shape)
        else:
            box_categories = self.box_categories
        box_emb = box_categories.unsqueeze(0).expand(BS, -1, -1, -1) + box_emb # [BS, T, O, d]

        oshape = box_emb.shape
        box_emb = box_emb.flatten(1,-2)
        box_emb, _  = self.attn(box_emb, None, None) # [BS, T, O,d]
        box_emb = box_emb.reshape(oshape)

        box_emb = self.obj2spatial(box_emb, None, box_tensors, H, W, t_avg_pooling=True) # [BS, T, H, W, d]
        return box_emb
