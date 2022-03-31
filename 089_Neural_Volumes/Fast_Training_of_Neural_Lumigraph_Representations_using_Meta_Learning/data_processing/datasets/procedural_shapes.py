"""
Represents 3D shape in single frame.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import open3d as o3d

import utils.math_utils as math_utils
import data_processing.components.pcd_io as pcd_io


class SpherePointCloud(Dataset):
    """
    Procedural sphere dataset.
    """

    def __init__(self, opt, radius=0.5):
        super().__init__()
        self.opt = opt
        self.radius = radius

        # Dummies.
        self.model_matrix = torch.from_numpy(np.eye(4, dtype=np.float32)).to(opt.device)
        self.coords = None
        self.normals = None
        self.colors = None

    def __len__(self):
        """
        Give it 200 iter per epoch to keep scale similar
        to typical meshes.
        """
        return 200

    def get_frame_coords(self, index: int):
        """
        Gets points of a single input PCD in the object space.
        """
        return self.coords

    def get_frame_normals(self, index: int):
        """
        Gets points of a single input PCD in the object space.
        """
        return self.normals

    def get_frame_colors(self, index: int):
        """
        Gets points of a single input PCD in the object space.
        """
        return self.colors

    @staticmethod
    def sample_unit_sphere(num_samples):
        """
        Uniformly sample sphere volume.
        """
        res = []
        num_res = 0
        # Uniform sphere sampling.
        while num_res < num_samples:
            pts = np.random.uniform(-1, 1, (num_samples, 3))
            pts_radius = np.linalg.norm(pts, axis=1)

            # Only accept inside sphere.
            is_valid = np.logical_and(pts_radius <= 1, pts_radius > 1e-5)
            pts_valid = pts[is_valid, :]
            if pts_valid.shape[0] > 0:
                res += [pts_valid]
                num_res += len(pts_valid)

        # Concat.
        res = np.concatenate(res, 0)[:num_samples, :]
        return res

    def __getitem__(self, idx):

        coords = self.sample_unit_sphere(self.opt.batch_size) * self.radius * 3
        distances = np.linalg.norm(coords, axis=-1)

        # SDF.
        sdf = distances - self.radius

        # Normals.
        normals = coords / distances[..., None].repeat(3, 1)
        #normals[sdf < 0,:] *= -1

        inputs = {
            'coords': torch.from_numpy(coords).float(),
            'model_matrix': self.model_matrix.cpu(),
        }
        gt = {
            'sdf': torch.from_numpy(sdf).float(),
            'normals': torch.from_numpy(normals).float(),
            'colors': torch.from_numpy(normals).float(),
        }

        return inputs, gt
