import torch
from . import ball_query_cuda


def safe_sqrt(x, eps=1e-12):
    return torch.sqrt(torch.clamp(x, eps))


class BallQueryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, radius):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        distance, count = ball_query_cuda.ball_query_forward(xyz1, xyz2, radius)
        ctx.save_for_backward(count, xyz1, xyz2)
        ctx.radius = radius
        return distance, count

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_dist = grad_outputs[0]
        count, xyz1, xyz2 = ctx.saved_tensors
        radius = ctx.radius
        grad_dist = grad_dist.contiguous()
        assert grad_dist.is_cuda, "Only support cuda currently."
        grad_xyz1, grad_xyz2 = ball_query_cuda.ball_query_backward(grad_dist, count, xyz1, xyz2, radius)
        return grad_xyz1, grad_xyz2, None


def ball_query(xyz1, xyz2, radius, transpose=True):
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    distance, count = BallQueryFunction.apply(xyz1, xyz2, radius)
    return distance, count
