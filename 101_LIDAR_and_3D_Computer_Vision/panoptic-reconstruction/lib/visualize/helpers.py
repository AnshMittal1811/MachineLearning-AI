import math
import os
from typing import Union, List, Tuple

import numpy as np
import torch

from . import io


def write_frustum(eight_point: Union[np.array, torch.Tensor], output_file: os.PathLike,
                  radius: float = 0.1, color: List = None):
    offset = [0, 0, 0]
    vertices = []
    faces = []
    colors = []

    if color is None:
        color = [128, 128, 128]

    edges = get_bbox_edges(eight_point)

    for edge in edges:
        edge_vertices, edge_faces = create_cylinder_mesh(radius, edge[0], edge[1])
        num_vertices = len(vertices)
        edge_color = [color for _ in edge_vertices]
        edge_vertices = [x + offset for x in edge_vertices]
        edge_faces = [x + num_vertices for x in edge_faces]

        vertices.extend(edge_vertices)
        faces.extend(edge_faces)
        colors.extend(edge_color)

    io.write_ply(vertices, colors, faces, output_file)


def get_bbox_edges(box_vertices) -> List[Tuple]:
    edges = [
        (box_vertices[0], box_vertices[1]),
        (box_vertices[1], box_vertices[2]),
        (box_vertices[2], box_vertices[3]),
        (box_vertices[3], box_vertices[0]),

        (box_vertices[4], box_vertices[5]),
        (box_vertices[5], box_vertices[6]),
        (box_vertices[6], box_vertices[7]),
        (box_vertices[7], box_vertices[4]),

        (box_vertices[0], box_vertices[4]),
        (box_vertices[1], box_vertices[5]),
        (box_vertices[2], box_vertices[6]),
        (box_vertices[3], box_vertices[7])
    ]

    return edges


def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10) -> Tuple[List, List]:
    def compute_length_vec3(vec3):
        return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

    def rotation(axis, angle):
        rot = np.eye(4)
        c = np.cos(-angle)
        s = np.sin(-angle)
        t = 1.0 - c
        axis /= compute_length_vec3(axis)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        rot[0, 0] = 1 + t * (x * x - 1)
        rot[0, 1] = z * s + t * x * y
        rot[0, 2] = -y * s + t * x * z
        rot[1, 0] = -z * s + t * x * y
        rot[1, 1] = 1 + t * (y * y - 1)
        rot[1, 2] = x * s + t * y * z
        rot[2, 0] = y * s + t * x * z
        rot[2, 1] = -x * s + t * y * z
        rot[2, 2] = 1 + t * (z * z - 1)
        return rot

    vertices = []
    faces = []
    diff = (p1 - p0).astype(np.float32)
    height = compute_length_vec3(diff)

    for i in range(stacks + 1):
        for i2 in range(slices):
            theta = i2 * 2.0 * math.pi / slices
            pos = np.array([radius * math.cos(theta), radius * math.sin(theta), height * i / stacks])
            vertices.append(pos)

    for i in range(stacks):
        for i2 in range(slices):
            i2p1 = math.fmod(i2 + 1, slices)
            faces.append(np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
            faces.append(
                np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32))

    transform = np.eye(4)
    va = np.array([0, 0, 1], dtype=np.float32)
    vb = diff
    vb /= compute_length_vec3(vb)
    axis = np.cross(vb, va)
    angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))

    if angle != 0:
        if compute_length_vec3(axis) == 0:
            dotx = va[0]

            if math.fabs(dotx) != 1.0:
                axis = np.array([1, 0, 0]) - dotx * va
            else:
                axis = np.array([0, 1, 0]) - va[1] * va

            axis /= compute_length_vec3(axis)

        transform = rotation(axis, -angle)

    transform[:3, 3] += p0
    vertices = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in vertices]
    vertices = [np.array([v[0], v[1], v[2]]) / v[3] for v in vertices]

    return vertices, faces
