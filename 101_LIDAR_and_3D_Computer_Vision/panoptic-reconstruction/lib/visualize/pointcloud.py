import os
from typing import Union, List, Tuple, Optional

import numpy as np
import torch

from . import io, utils


def write_pointcloud(points: Union[np.array, torch.Tensor], colors: Union[np.array, torch.Tensor, List, Tuple],
                     output_file: os.PathLike) -> None:
    io.write_ply(points, colors, None, output_file)


def write_semantic_pointcloud(points: Union[np.array, torch.Tensor], labels: Union[np.array, torch.Tensor],
                              output_file: os.PathLike, color_palette=None) -> None:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    colors = utils.lookup_colors(labels, color_palette)
    write_pointcloud(points, colors, output_file)
