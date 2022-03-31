import torch
from . import fps_cuda


def select_points(points, index):
    """Gather xyz of centroids according to indices

    Args:
        points: (batch_size, channels, num_points)
        index: (batch_size, num_centroids)

    Returns:
        new_xyz (torch.Tensor): (batch_size, channels, num_centroids)

    """
    batch_size = points.size(0)
    channels = points.size(1)
    num_centroids = index.size(1)
    index_expand = index.unsqueeze(1).expand(batch_size, channels, num_centroids)
    return points.gather(2, index_expand)


class FarthestPointSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, num_centroids):
        index = fps_cuda.farthest_point_sample(points.contiguous(), num_centroids)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None


def farthest_point_sample(points, num_centroids, transpose=True):
    """Farthest point sample

    Args:
        points (torch.Tensor): (batch_size, 3, num_points)
        num_centroids (int): the number of centroids to sample
        transpose (bool): whether to transpose points

    Returns:
        index (torch.Tensor): (batch_size, num_centroids), sample indices of centroids.

    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if transpose:
        points = points.transpose(1, 2)
    points = points.contiguous()
    return FarthestPointSampleFunction.apply(points, num_centroids)


@torch.no_grad()
def farthest_point_sample_wrapper(xyz, nb_pts):
    xyz = torch.tensor(xyz, dtype=torch.float32).cuda(non_blocking=True)
    fps_idx = farthest_point_sample(xyz, min(nb_pts, xyz.size(0)), transpose=False).squeeze(0)
    return xyz[fps_idx].cpu().numpy()
