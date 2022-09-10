import os
from typing import Union, Tuple, Optional

import numpy as np
import torch

import marching_cubes as mc

from lib.utils.transform import coords_multiplication
from . import io, utils


def write_distance_field(distance_field: Union[np.array, torch.Tensor], labels: Optional[Union[np.array, torch.Tensor]],
                         output_file: os.PathLike, iso_value: float = 1.0, truncation: float = 3.0,
                         color_palette=None, transform=None) -> None:
    if isinstance(distance_field, torch.Tensor):
        distance_field = distance_field.detach().cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if labels is not None:
        vertices, colors, triangles = get_mesh_with_semantics(distance_field, labels, iso_value, truncation,
                                                              color_palette)
    else:
        vertices, triangles = get_mesh(distance_field, iso_value, truncation)
        colors = None

    if transform is not None:
        if isinstance(transform, torch.Tensor):
            transform = transform.detach().cpu().numpy()

        vertices = coords_multiplication(transform, vertices)

    io.write_ply(vertices, colors, triangles, output_file)


def get_mesh(distance_field: np.array, iso_value: float = 1.0, truncation: float = 3.0) -> Tuple[np.array, np.array]:
    vertices, triangles = mc.marching_cubes(distance_field, iso_value, truncation)
    return vertices, triangles


def get_mesh_with_semantics(distance_field: np.array, labels: np.array, iso_value: float = 1.0, truncation: float = 3.0,
                            color_palette=None) -> Tuple[np.array, np.array, np.array]:
    labels = labels.astype(np.uint32)
    color_volume = utils.lookup_colors(labels, color_palette)
    vertices, colors, triangles = get_mesh_with_colors(distance_field, color_volume, iso_value, truncation)

    return vertices, colors, triangles


def get_mesh_with_colors(distance_field: np.array, colors: np.array, iso_value: float = 1.0,
                         truncation: float = 3.0) -> Tuple[np.array, np.array, np.array]:
    vertices, triangles = mc.marching_cubes_color(distance_field, colors, iso_value, truncation)
    colors = vertices[..., 3:]
    vertices = vertices[..., :3]

    return vertices, colors, triangles
