import torch
from . import nn_distance_cuda


def safe_sqrt(x, eps=1e-12):
    return torch.sqrt(torch.clamp(x, eps))


class NNDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        dist1, idx1 = nn_distance_cuda.nn_distance_forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1)
        return dist1, idx1

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_dist1 = grad_outputs[0]
        xyz1, xyz2, idx1 = ctx.saved_tensors
        grad_dist1 = grad_dist1.contiguous()
        assert grad_dist1.is_cuda, "Only support cuda currently."
        grad_xyz1, grad_xyz2 = nn_distance_cuda.nn_distance_backward(grad_dist1, xyz1, xyz2, idx1)
        return grad_xyz1, grad_xyz2


def nn_distance(xyz1, xyz2, transpose=True, sqrt=False, eps=1e-12):
    """NN distance

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n2)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.
        sqrt (bool): whether to square root distance
        eps (float): to safely sqrt

    Returns:
        dist1 (torch.Tensor): (b, n1)
        idx1 (torch.Tensor): (b, n1)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    dist1, idx1 = NNDistanceFunction.apply(xyz1, xyz2)
    if sqrt:
        dist1 = safe_sqrt(dist1, eps)
    return dist1, idx1


@torch.no_grad()
def chamfer_distance_wrapper(xyz1, xyz2):
    """Chamfer distance wrapper for numpy array

    Args:
        xyz1 (numpy.ndarray): (N1, 3)
        xyz2 (numpy.ndarray): (N2, 3)

    Returns:
        dist1 (numpy.ndarray): (N1,)
        dist2 (numpy.ndarray): (N2,)

    """
    xyz1 = torch.tensor(xyz1, dtype=torch.float32).cuda(non_blocking=True)
    xyz2 = torch.tensor(xyz2, dtype=torch.float32).cuda(non_blocking=True)
    dist1, idx1 = nn_distance(xyz1, xyz2, transpose=False, sqrt=True)
    dist2, idx2 = nn_distance(xyz2, xyz1, transpose=False, sqrt=True)
    return dist1[0].cpu().numpy(), dist2[0].cpu().numpy()
