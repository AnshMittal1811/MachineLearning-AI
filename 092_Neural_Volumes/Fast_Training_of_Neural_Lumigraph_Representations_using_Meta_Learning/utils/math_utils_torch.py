"""
Utilities for geometry etc.
"""

import torch


def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    #res = torch.matmul(vectors4, matrix.T)
    res = (matrix @ vectors4[..., None])[..., 0]
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / torch.norm(vectors, dim=-1, keepdim=True).detach()


def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)


def glFrustrum(b, t, l, r, n, f):
    """
    Get OpenGL projection matrix.
    """
    zeros = torch.zeros_like(b).to(b.device)
    ones = torch.ones_like(b).to(b.device)

    return torch.stack([
        torch.stack([2 * n / (r - l), zeros, (r + l) / (r - l), zeros], -1),
        torch.stack([zeros, 2 * n / (t - b), (t + b) / (t - b), zeros], -1),
        torch.stack([zeros, zeros, -(f + n) / (f - n), -2 * f * n / (f - n)], -1),
        torch.stack([zeros, zeros, -ones, zeros], -1)
    ], -2)


def quaternion_to_matrix(q: torch.Tensor, assume_normalized=True):
    """
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    """
    s = 1 if assume_normalized else 1 / torch.sum(q * q, -1)
    q_i = q[..., 0:1]
    q_j = q[..., 1:2]
    q_k = q[..., 2:3]
    q_r = q[..., 3:4]
    return torch.stack([
        torch.stack([1 - 2 * s * (q_j ** 2 + q_k ** 2),
                     2 * s * (q_i * q_j - q_k * q_r),
                     2 * s * (q_i * q_k + q_j * q_r)], -1),
        torch.stack([2 * s * (q_i * q_j + q_k * q_r),
                     1 - 2 * s * (q_i ** 2 + q_k ** 2),
                     2 * s * (q_j * q_k - q_i * q_r)], -1),
        torch.stack([2 * s * (q_i * q_k - q_j * q_r),
                     2 * s * (q_j * q_k + q_i * q_r),
                     1 - 2 * s * (q_i ** 2 + q_j ** 2)], -1),
    ], -2)
