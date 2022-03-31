import torch
from . import group_points_cuda


class GroupPointsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, index):
        ctx.save_for_backward(index)
        ctx.num_points = points.size(2)
        group_points = group_points_cuda.group_points_forward(points, index)
        return group_points

    @staticmethod
    def backward(ctx, *grad_output):
        index = ctx.saved_tensors[0]
        grad_input = group_points_cuda.group_points_backward(grad_output[0], index, ctx.num_points)
        return grad_input, None


def group_points(points, index):
    """Gather points by index

    Args:
        points (torch.Tensor): (batch_size, channels, num_points)
        index (torch.Tensor): (batch_size, num_centroids, num_neighbours), indices of neighbours of each centroid.

    Returns:
        group_points (torch.Tensor): (batch_size, channels, num_centroids, num_neighbours), grouped points.

    """
    return GroupPointsFunction.apply(points, index)
