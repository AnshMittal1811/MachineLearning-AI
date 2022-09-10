import math
from typing import Tuple

import numpy as np
import torch


def frustum2planes(frustum):
    planes = {}
    # normal towards inside
    # near
    a = frustum[3] - frustum[0]
    b = frustum[1] - frustum[0]
    normal = np.cross(a, b)
    d = -np.dot(normal, frustum[0])
    planes["near"] = np.array([normal[0], normal[1], normal[2], d])

    # far
    a = frustum[5] - frustum[4]
    b = frustum[7] - frustum[4]
    normal = np.cross(a, b)
    d = -np.dot(normal, frustum[4])
    planes["far"] = np.array([normal[0], normal[1], normal[2], d])

    # left
    a = frustum[5] - frustum[1]
    b = frustum[0] - frustum[1]
    normal = np.cross(a, b)
    d = -np.dot(normal, frustum[1])
    planes["left"] = np.array([normal[0], normal[1], normal[2], d])

    # right
    a = frustum[3] - frustum[2]
    b = frustum[6] - frustum[2]
    normal = np.cross(a, b)
    d = -np.dot(normal, frustum[2])
    planes["right"] = np.array([normal[0], normal[1], normal[2], d])

    # top
    a = frustum[4] - frustum[0]
    b = frustum[3] - frustum[0]
    normal = np.cross(a, b)
    d = -np.dot(normal, frustum[0])
    planes["top"] = np.array([normal[0], normal[1], normal[2], d])

    # bottom
    a = frustum[2] - frustum[1]
    b = frustum[5] - frustum[1]
    normal = np.cross(a, b)
    d = -np.dot(normal, frustum[1])
    planes["bottom"] = np.array([normal[0], normal[1], normal[2], d])

    return planes


def frustum_culling(points, frustum):
    frustum_planes = frustum2planes(frustum)
    points = np.concatenate([points, np.ones((len(points), 1))], 1)
    flags = np.ones(len(points))
    for key, plane in frustum_planes.items():
        flag = np.dot(points, plane) >= 0
        flags = np.logical_and(flags, flag)

    return points[flags][:, :3]


def frustum_transform(frustum, transform):
    eight_points = np.concatenate([frustum, np.ones((8, 1))], 1).transpose()
    frustum = np.dot(transform, eight_points).transpose()
    return frustum[:, :3]


def generate_frustum(image_size, intrinsic_inv, depth_min, depth_max, transform=None):
    x = image_size[0]
    y = image_size[1]

    eight_points = np.array([[0 * depth_min, 0 * depth_min, depth_min, 1.0],
                             [0 * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, 0 * depth_min, depth_min, 1.0],
                             [0 * depth_max, 0 * depth_max, depth_max, 1.0],
                             [0 * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, 0 * depth_max, depth_max, 1.0]]).transpose()

    frustum = np.dot(intrinsic_inv, eight_points)

    if transform is not None:
        frustum = np.dot(transform, frustum)

    frustum = frustum.transpose()

    return frustum[:, :3]


def generate_frustum_volume(frustum, voxel_size):
    max_x = np.max(frustum[:, 0]) / voxel_size
    max_y = np.max(frustum[:, 1]) / voxel_size
    max_z = np.max(frustum[:, 2]) / voxel_size
    min_x = np.min(frustum[:, 0]) / voxel_size
    min_y = np.min(frustum[:, 1]) / voxel_size
    min_z = np.min(frustum[:, 2]) / voxel_size

    dim_x = math.ceil(max_x - min_x)
    dim_y = math.ceil(max_y - min_y)
    dim_z = math.ceil(max_z - min_z)

    camera2frustum = np.array([[1.0 / voxel_size, 0, 0, -min_x],
                               [0, 1.0 / voxel_size, 0, -min_y],
                               [0, 0, 1.0 / voxel_size, -min_z],
                               [0, 0, 0, 1.0]])

    return (dim_x, dim_y, dim_z), camera2frustum


def compute_camera2frustum_transform(intrinsic: torch.Tensor, image_size: Tuple, depth_min: float, depth_max: float,
                                     voxel_size: float) -> torch.Tensor:
    frustum = generate_frustum(image_size, torch.inverse(intrinsic), depth_min, depth_max)
    _, camera2frustum = generate_frustum_volume(frustum, voxel_size)
    camera2frustum = torch.from_numpy(camera2frustum).float()

    return camera2frustum
