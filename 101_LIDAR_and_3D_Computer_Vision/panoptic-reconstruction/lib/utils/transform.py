import numpy as np
import torch


def coords_multiplication(matrix, points):
    """
    matrix: 4x4
    points: nx3
    """

    if isinstance(matrix, torch.Tensor):
        device = torch.device("cuda:0" if matrix.get_device() != -1 else "cpu")
        points = torch.cat([points.t(), torch.ones((1, points.shape[0]), device=device)])
        return torch.mm(matrix, points).t()[:, :3]
    elif isinstance(matrix, np.ndarray):
        points = np.concatenate([np.transpose(points), np.ones((1, points.shape[0]))])
        return np.transpose(np.dot(matrix, points))[:, :3]
