# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import random_rotations, rotation_6d_to_matrix, matrix_to_rotation_6d, \
    matrix_to_axis_angle, euler_angles_to_matrix
import pytorch3d.transforms.rotation_conversions as rot_cvt
from pytorch3d.transforms import Transform3d, Rotate, Translate, Scale


def scale_matrix(scale, homo=True):
    """
    :param scale: (..., 3)
    :return: scale matrix (..., 4, 4)
    """
    dims = scale.size()[0:-1]
    one_dims = [1,] * len(dims)
    device = scale.device
    if scale.size(-1) == 1:
        scale = scale.expand(*dims, 3)
    mat = torch.diag_embed(scale, dim1=-2, dim2=-1)
    if homo:
        mat = rt_to_homo(mat)
    return mat


def se3_to_matrix(param: torch.Tensor):
    """
    :param param: tensor in shape of (..., 10) rotation param (6) + translation (3) + scale (1)
    :return: transformation matrix in shape of (N, 4, 4) sR+t
    """
    rot6d, trans, scale = torch.split(param, [6, 3, 3], dim=-1)
    rot = rotation_6d_to_matrix(rot6d)  # N, 3, 3
    mat = rt_to_homo(rot, trans, scale)    
    return mat


def matrix_to_se3(mat: torch.Tensor) -> torch.Tensor:
    """
    :param mat: transformation matrix in shape of (N, 4, 4)
    :return: tensor in shape of (N, 9) rotation param (6) + translation (3)
    """
    rot, trans, scale = homo_to_rt(mat)
    rot = matrix_to_rotation_6d(rot)
    se3 = torch.cat([rot, trans, scale], dim=-1)
    return se3


def mat_to_scale_rot(mat):
    """s*R to s, R
    
    Args:
        mat ( ): (..., 3, 3)
    Returns:
        scale: (..., 3)
        rot: (..., 3, 3)
    """
    sq = torch.matmul(mat, mat.transpose(-1, -2))
    scale_flat = torch.sqrt(torch.diagonal(sq, dim1=-1, dim2=-2))  # (..., 3)
    scale_inv = scale_matrix(1/scale_flat, homo=False)
    rot = torch.matmul(mat, scale_inv)
    return rot, scale_flat



def se3_to_axis_angle_t(param: torch.Tensor):
    """
    :param param: tensor in shape of (N, 9) rotation param (6) + translation (3)
    :return: rot in axis-angle (N, 3), translation (N, 3), scale (N, 1)
    """
    rot6d, trans, scale = torch.split(param, [6, 3, 3], dim=-1)
    axisang = matrix_to_axis_angle(rotation_6d_to_matrix(rot6d))
    return axisang, trans, scale


def axis_angle_t_to_matrix(axisang=None, t=None, s=None, homo=True):
    """
    :param axisang: (N, 3)
    :param t: (N, 3)
    :return: (N, 4, 4)
    """
    if axisang is None:
        axisang = torch.zeros_like(t)
    if t is None:
        t = torch.zeros_like(axisang)
    rot = rot_cvt.axis_angle_to_matrix(axisang)
    if homo:
        return rt_to_homo(rot, t, s)
    else:
        return rot

def matrix_to_axis_angle_t(mat: torch.Tensor):
    r, t, s = homo_to_rt(mat)
    return matrix_to_axis_angle(r), t, s


def azel_to_rot(azel, homo=False, t=None):
    zeros = torch.zeros(list(azel.size())[:-1] +[1]).to(azel)
    euler_angles = torch.cat([azel, zeros], dim=-1)
    rot = euler_angles_to_matrix(euler_angles, 'YXY')
    if homo:
        rot = rt_to_homo(rot, t)
    return rot


def rt_to_homo(rot, t=None, s=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :param s: (..., 1)
    :return: (N, 4, 4) [R, t; 0, 1] sRX + t
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] += 1
    mat = torch.cat([mat, zeros], dim=-2)
    if s is not None:
        s = scale_matrix(s)
        mat = torch.matmul(mat, s)

    return mat


def homo_to_rt(mat):
    """
    :param (N, 4, 4) [R, t; 0, 1]
    :return: rot: (N, 3, 3), t: (N, 3), s: (N, 1)
    """
    mat, _ = torch.split(mat, [3, mat.size(-2) - 3], dim=-2)
    rot_scale, trans = torch.split(mat, [3, 1], dim=-1)
    rot, scale = mat_to_scale_rot(rot_scale)

    trans = trans.squeeze(-1)
    return rot, trans, scale


def random_se3(N=1, device='cpu'):
    rot = random_rotations(N, device=device)
    trans = torch.zeros([N, 3], device=device)
    scale = torch.ones([N, 3], device=device)
    se3 = matrix_to_se3(rt_to_homo(rot, trans, scale))
    return se3
    

def jitter_se3(se3, rot_stddev, t_stddev, s_stddev=0):
    N = se3.size(0)
    device = se3.device

    jitter_t = torch.rand([N, 3], device=device) * t_stddev * 2 - t_stddev
    # jitter_t = jitter_t.clamp(-t_stddev * 2, t_stddev * 2)

    jitter_axiang = torch.rand([N, 3], device=device) *  rot_stddev * 2 - rot_stddev
    # jitter_axiang = jitter_axiang.clamp(-rot_stddev * 2, rot_stddev * 2)

    jitter_scale = torch.exp(torch.rand([N, 1], device=device) *  s_stddev * 2 - s_stddev)
    jitter_scale = jitter_scale.repeat(1, 3)

    mat = se3_to_matrix(se3)
    delta_mat = axis_angle_t_to_matrix(jitter_axiang, jitter_t, jitter_scale)
    dst_mat = torch.matmul(mat, delta_mat)
    dst_se3 = matrix_to_se3(dst_mat)
    return dst_se3, delta_mat


def inverse_rt(se3=None, mat=None, return_mat=False):
    """
    [R, t] --> [R.T, -R.T + t]
    :param se3:
    :param mat:
    :param return_mat:
    :return:
    """
    if mat is None:
        mat = se3_to_matrix(se3)

    rot, trans, scale = homo_to_rt(mat)
    inv_scale = scale_matrix(1 / scale, homo=False)
    inv_mat = rt_to_homo(rot.transpose(-1, -2) @ inv_scale,  
        (-rot.transpose(-1, -2) @ inv_scale) @ trans.unsqueeze(-1))
    if return_mat:
        return inv_mat
    else:
        return matrix_to_se3(inv_mat)


def rt_to_transform(se3=None, mat=None, ) -> Transform3d:
    """
    :param se3:
    :param mat: 
    :return: Transform3d: sRX + t . rot.compose(t)
    """
    if mat is None:
        mat = se3_to_matrix(se3)
    device = mat.device
    rot, trans, scale = homo_to_rt(mat)
    rot = Rotate(rot.transpose(1, 2), device=device)
    trans = Translate(trans, device=device)
    scale = Scale(scale, device=device)
    return scale.compose(rot.compose(trans))
    

def compose_se3(se_a, se_b, return_mat=False):
    mat_a = se3_to_matrix(se_a)
    mat_b = se3_to_matrix(se_b)
    mat = torch.matmul(mat_a, mat_b)
    if return_mat:
        return mat
    else:
        return matrix_to_se3(mat)

def trans3d(mat, points):
    N, P, _  = points.size()
    device = points.device
    ones = torch.ones(N, P, 1)
    pts = torch.cat([points, torch.ones(N, P, 1, device=device)])
    pts = (mat @ pts.transpose(-1, -2)).transpose(-1, -2)
    pts = pts[..., :-3] / pts[..., -3:]
    return pts
    