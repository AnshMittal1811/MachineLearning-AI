import torch
from . import chamfer_cuda


def safe_sqrt(x, eps=1e-12):
    return torch.sqrt(torch.clamp(x, eps))


class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        dist1, idx1, dist2, idx2 = chamfer_cuda.chamfer_forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_dist1, grad_dist2 = grad_outputs[0:2]
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_dist1 = grad_dist1.contiguous()
        grad_dist2 = grad_dist2.contiguous()
        assert grad_dist1.is_cuda and grad_dist2.is_cuda, "Only support cuda currently."
        grad_xyz1, grad_xyz2 = chamfer_cuda.chamfer_backward(grad_dist1, grad_dist2, xyz1, xyz2, idx1, idx2)
        return grad_xyz1, grad_xyz2


def chamfer_distance(xyz1, xyz2, transpose=True, sqrt=False, eps=1e-12, return_indices=False):
    """Chamfer distance

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.
        sqrt (bool): whether to square root distance
        eps (float): to safely sqrt
        return_indices (bool): whether to return indices

    Returns:
        dist1 (torch.Tensor): (b, n1)
        dist2 (torch.Tensor): (b, n2)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    dist1, dist2, idx1, idx2 = ChamferDistanceFunction.apply(xyz1, xyz2)
    if sqrt:
        dist1 = safe_sqrt(dist1, eps)
        dist2 = safe_sqrt(dist2, eps)
    if return_indices:
        return dist1, dist2, idx1, idx2
    else:
        return dist1, dist2


def nn_distance(xyz1, xyz2, transpose=True):
    """The interface to infer rather than train"""
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2).contiguous()
        xyz2 = xyz2.transpose(1, 2).contiguous()
    return chamfer_cuda.chamfer_forward(xyz1, xyz2)
