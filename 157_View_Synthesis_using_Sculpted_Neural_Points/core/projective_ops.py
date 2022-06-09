import torch
import torch.nn.functional as F
from opt_einsum import contract


import subprocess
import time


def coords_grid(d):
    ht, wd = d.shape[-2], d.shape[-1]
    y, x = torch.meshgrid(torch.arange(ht), torch.arange(wd))

    # trick to expand / reshape
    x = x.to(d.device).float() + torch.zeros_like(d)
    y = y.to(d.device).float() + torch.zeros_like(d)

    return torch.stack([x, y, torch.ones_like(d), d], -1)

def projective_transform(Ps, disps, intrinsics, ii, jj, ii_reduced=None):
    # 3x3 -> 4x4 intrinsics matrix
    Ks = torch.zeros_like(Ps)
    Ks[...,:3,:3] = intrinsics
    Ks[..., 3, 3] = 1.0

    # 4x4 projection matricies
    Pij = Ks[:, jj] @ Ps[:, jj] @ Ps[:, ii].inverse() @ Ks[:, ii].inverse()
    # Pji = Ks[:, ii] @ Ps[:, ii] @ Ps[:, jj].inverse() @ Ks[:, jj].inverse()

    # print(ii)
    x0 = coords_grid(disps[:,ii if ii_reduced is None else ii_reduced])
    x1 = contract('ijkh,ij...h->ij...k', Pij, x0)

    return x1 / x1[..., [2]]


def backproject(Ps, intrinsics, disps):
    Ks = torch.zeros_like(Ps)
    Ks[...,:3,:3] = intrinsics
    Ks[..., 3, 3] = 1.0

    P = Ps.inverse() @ Ks.inverse()
    x0 = coords_grid(disps)

    X = contract('aij,a...j->a...i', P, x0)
    return X[..., :3] / X[..., [3]]


