# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


# def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
#     # for debug and test only,
#     # need to use cuda version instead
#     """
#     N_ -> Batch size
#     S_ -> W*H over each spatial resolution, from higher to lower
#     M_ -> Num heads
#     D_ -> Rest of channels after split to have n_heads
#     Lq_ -> Num embeddings (300 or positive matched embeddings)
#     L_ -> Num of input level resolutions
#     P_ -> Num of sampling points for each operation
#     """
#
#     N_, S_, M_, D_ = value.shape
#     _, Lq_, M_, L_, P_, _ = sampling_locations.shape
#
#     # Recovers to a list of the tensors separated for each resolution
#     value_list = value.split([H_ * W_ * T_ for H_, W_, T_ in value_spatial_shapes], dim=1)
#     sampling_grids = 2 * sampling_locations - 1
#     sampling_value_list = []
#
#     # For each spatial resolution that we have
#     for lid_, (H_, W_, T_) in enumerate(value_spatial_shapes):
#         # 1) Pick values of that spatial resolution
#         # 2) Flatten [Num_heads, Num_channels] to single dimension (recovers 256 channels)
#         # 3) Puts channels on the first axis and resolution flattened to the last one
#         # 4) Splits again channel axis to have 32 channels again and passing num_heads to the batch position
#
#         # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
#         # value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, T_, H_, W_)
#         value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, T_*H_, W_)
#
#         # 1)Pick all available sampling points from that spatial resolution
#         # 2) Swap num_embeddings with num_heads axis, so this passes to be the first one
#         # 3) Flatten batch_size and Num_heads to single axis
#
#         # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
#         sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
#
#         # Returns [batch_size*num_heads, 32, num_embds, num_sample_points]
#         # N_*M_, D_, Lq_, P_
#         sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
#         sampling_value_list.append(sampling_value_l_)
#
#     # 1) Swap num_embds axis with num_heads axis
#     # 2) Reshape to [Batch_size*Num_heads, 1, Num_embds, Num_lvl_res * sampling_points] (Simply Flatten num_lvl_res & num_sampling_points)
#     # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
#     attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
#
#     # 1) List sampling_value_list is the result of the sampling points in each resolution shape -> [bs*num_heads, 32, num_embds, sampling_points]
#     # 2) Stack to add the level dimension in between num_embds & samplings points -> [bs*num_heads, 32, num_embds, num_res_lvls, sampling_points]
#     # 3) Flatten num_res_lvls & sampling_points to single dimension -> [bs*num_heads, 32, num_embds, num_res_lvls*sampling_points]
#     # 4) Dot product by attention weights of shape [bs*num_heads, 1, num_embds, num_res_lvls*sampling_points]
#         # resulting in  -> [bs*num_heads, 32, num_embds, num_res_lvls*sampling_points]
#     # 5) Sum over the last axis which is num_res_lvls*sampling_points  -> [bs*num_heads, 32, num_embds]
#     # 6) Reshape to [bs, num_channels*num_heads (256),  Num_embds]
#     # 7) Put channels last axis
#     output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
#     return output.transpose(1, 2).contiguous()


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()