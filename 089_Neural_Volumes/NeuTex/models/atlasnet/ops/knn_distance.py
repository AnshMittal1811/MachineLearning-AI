import torch
from . import knn_distance_cuda


class KNNDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_xyz, key_xyz, k):
        index, distance = knn_distance_cuda.knn_distance(query_xyz, key_xyz, k)
        return index, distance

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None


def knn_distance(query, key, k, transpose=True):
    """For each point in query set, find its distances to k nearest neighbors in key set.

    Args:
        query: (B, 3, N1), xyz of the query points.
        key: (B, 3, N2), xyz of the key points.
        k (int): K nearest neighbor
        transpose (bool): whether to transpose xyz

    Returns:
        index: (B, N1, K), indices of these neighbors in the key.
        distance: (B, N1, K), distance to the k nearest neighbors in the key.

    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    query = query.contiguous()
    key = key.contiguous()
    index, distance = KNNDistanceFunction.apply(query, key, k)
    return index, distance
