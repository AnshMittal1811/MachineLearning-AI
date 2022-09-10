import os
from typing import Union, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

from plyfile import PlyData


def read_ply(ply_file):
    with open(ply_file, "rb") as file:
        ply_data = PlyData.read(file)

    points = []
    colors = []
    indices = []

    for x, y, z, r, g, b in ply_data["vertex"]:
        points.append([x, y, z])
        colors.append([r, g, b])

    for face in ply_data["face"]:
        indices.append([face[0][0], face[0][1], face[0][2]])

    points = np.array(points)
    colors = np.array(colors)
    indices = np.array(indices)

    return points, indices, colors


def write_ply(vertices: Union[np.array, torch.Tensor], colors: Union[np.array, torch.Tensor, List, Tuple],
              faces: Union[np.array, torch.Tensor], output_file: os.PathLike) -> None:
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()

    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    if colors is not None:
        if isinstance(colors, list) or isinstance(colors, tuple):
            colors = np.ones_like(vertices) * np.array(colors)

    if faces is None:
        faces = []

    with open(output_file, "w") as file:
        file.write("ply \n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices):d}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")

        if colors is not None:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")

        if faces is not None:
            file.write(f"element face {len(faces):d}\n")
            file.write("property list uchar uint vertex_indices\n")
        file.write("end_header\n")

        if colors is not None:
            for vertex, color in zip(vertices, colors):
                file.write(f"{vertex[0]:f} {vertex[1]:f} {vertex[2]:f} ")
                file.write(f"{int(color[0]):d} {int(color[1]):d} {int(color[2]):d}\n")
        else:
            for vertex in vertices:
                file.write(f"{vertex[0]:f} {vertex[1]:f} {vertex[2]:f}\n")

        for face in faces:
            file.write(f"3 {face[0]:d} {face[1]:d} {face[2]:d}\n")


def write_image(image: Union[np.array, torch.Tensor], output_file: os.PathLike, kwargs=None) -> None:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    plt.imsave(output_file, image, **kwargs)
