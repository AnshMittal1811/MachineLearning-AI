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
import utils.common_utils as common_utils

"""
Model matrix priority:
1) Save model_matrix.npy
2) PCD stats in scene_stats.npy
3) Camera intersection in scene_stats.npy
4) PCD stats from loaded PCD
5) Identity matrix
"""


class PointCloudVideo(Dataset):
    def __init__(self, dataset_path: Path, opt):
        super().__init__()
        self.opt = opt

        # Initialized empty for typing.
        self.coords = np.zeros([0, 3], np.float32)
        self.normals = np.zeros([0, 3], np.float32)
        self.colors = np.zeros([0, 3], np.float32)
        self.frame_indices = np.zeros([0, ], int)
        # Offsets of individual input Frames
        # Has start and end indices for each frame.
        self.frame_offsets = np.zeros([0, ], int)

        # An optional matrix to scale from the input PCD to
        # the actual original dataset - for benchmarks.
        self.m_pcd_to_original = np.eye(4, dtype=np.float32)
        self.scene_stats = self.load_scene_stats(dataset_path)

        # Try to load model matrix.
        self.model_matrix_mode = None
        self.model_matrix = self.select_model_matrix(dataset_path)

        # Read PCDs if asked or if needed to compute the model matrix.
        if self.opt.load_pcd or self.model_matrix is None:
            self.load_dataset(dataset_path)
        else:
            # We cannot use pcd if it is not loaded.
            self.opt.use_pcd = False

        if self.model_matrix is None:
            self.model_matrix = torch.from_numpy(np.eye(4, dtype=np.float32)).to(opt.device)

    def load_dataset(self, dataset_path):
        """
        Loads the dataset.
        """

        # Load raw daset.
        if not self._load_all_frames_data(dataset_path):
            self.opt.use_pcd = False
            return

        # Reshape point cloud such that it lies in a radius=1-eps sphere
        if self.model_matrix is None:
            coord_min = np.min(self.coords, axis=0)
            coord_max = np.max(self.coords, axis=0)
            pcd_center = (coord_min + coord_max) / 2
            translation = -pcd_center
            diff_center = np.linalg.norm(self.coords + translation, axis=1)
            max_radius = diff_center.max()
            scale = self.opt.work_radius / max_radius
            norm_matrix = np.eye(4, dtype=np.float32)
            norm_matrix[0, 0] = scale
            norm_matrix[1, 1] = scale
            norm_matrix[2, 2] = scale
            norm_matrix[:3, 3] = translation * scale

            # Model matrix will convert the coords back to the input shape.
            self.model_matrix = torch.from_numpy(np.linalg.inv(norm_matrix)).to(self.opt.device)
            self.model_matrix_mode = 'cache'
        else:
            norm_matrix = np.linalg.inv(self.model_matrix.cpu().numpy())

        # Apply coord transform.
        self.coords = math_utils.transform_points(norm_matrix, self.coords, return_euclidean=True)
        # Also transform normals.
        self.normals = math_utils.transform_normals(norm_matrix, self.normals)
        # Save model matrix.
        self.save_model_matrix(dataset_path, self.model_matrix, overwrite=(self.model_matrix_mode == 'cache'))

        # Mark NaN normals as zero and not use them for supervision.
        self.normals[np.any(np.isnan(self.normals), 1), :] = 0

        # Normalize colors as with torchvision.transforms.Normalize(mean=0.5, std=0.5)
        if self.colors is not None:
            # The range is [0, 1] -> [-1, 1]
            self.colors = (self.colors - 0.5) / 0.5

    def _load_all_frames_data(self, dataset_path: Path) -> bool:
        """
        Loads all PCD frames in given path.
        """
        frames = []
        if self.opt.dataset_type == 'sinesdf_static':
            # Just single frame.
            frames = [self._load_frame_data(dataset_path)]
        else:
            # Load all frames.
            frame_dirs = sorted([x for x in dataset_path.iterdir() if x.is_dir()])
            for i, frame_dir in enumerate(frame_dirs):
                print(f'[{i}/{len(frame_dirs)}] Loading PCD frame {frame_dir}...')
                frames += [self._load_frame_data(frame_dir)]

        # Store pcd->original transform.
        if len(frames) > 0 and frames[0] is not None and 'm_pcd_to_original' in frames[0]:
            self.m_pcd_to_original = frames[0]['m_pcd_to_original']

        # Merge all frames.
        all_coords = []
        all_normals = []
        all_colors = []
        all_frame_indices = []
        all_frame_offsets = [0, ]
        for i, frame in enumerate(frames):
            if frame is None:
                all_frame_offsets += [all_frame_offsets[-1] + 0]
                continue

            all_coords += [frame['points']]
            all_normals += [frame['normals']]
            all_colors += [frame['colors']]
            all_frame_indices += [np.zeros((frame['points'].shape[0],), int) + i]
            all_frame_offsets += [all_frame_offsets[-1] + frame['points'].shape[0]]

        total_points = all_frame_offsets[-1]
        if total_points <= 0:
            return False

        # Merge all pcds.
        self.coords = np.concatenate(all_coords, 0)
        self.normals = np.concatenate(all_normals, 0)
        self.colors = np.concatenate(all_colors, 0)
        self.frame_indices = np.concatenate(all_frame_indices, 0)
        self.frame_offsets = np.array(all_frame_offsets)

        print(f'Loaded {self.coords.shape[0]} 3D points.')

        return True

    def _load_frame_data(self, frame_path: Path) -> dict:
        """
        Loads all PCDs in given path.
        """
        # Try per-frame global PCD.
        world_pcd = frame_path / 'merged' / 'point_cloud_world.ply'
        if world_pcd.is_file():
            # Use alternative PCD.
            print(f'\tLoading alternative world PCD from {world_pcd}...')
            points, normals, colors, m_pcd_to_original = pcd_io.load_point_cloud(world_pcd)
            offsets = [0, points.shape[0]]
            return {
                'points': points,
                'normals': normals,
                'colors': colors,
                'offsets': offsets,
                'm_pcd_to_original': m_pcd_to_original,
            }

        return pcd_io.load_all_point_clouds(frame_path)

    @property
    def num_frames(self) -> int:
        """
        Number of inputs PCDs.
        """
        return len(self.frame_offsets) - 1

    def get_frame_coords(self, index: int):
        """
        Gets points of a single input PCD in the object space.
        """
        if self.coords.shape[0] == 0:
            return self.coords
        return self.coords[self.frame_offsets[index]:self.frame_offsets[index + 1], ...]

    def get_frame_normals(self, index: int):
        """
        Gets points of a single input PCD in the object space.
        """
        if self.coords.shape[0] == 0:
            return self.normals
        return self.normals[self.frame_offsets[index]:self.frame_offsets[index + 1], ...]

    def get_frame_colors(self, index: int):
        """
        Gets points of a single input PCD in the object space.
        """
        if self.coords.shape[0] == 0:
            return self.colors
        return self.colors[self.frame_offsets[index]:self.frame_offsets[index + 1], ...]

    def load_scene_stats(self, dataset_path: Path) -> np.array:
        """
        Tries to load scene statistics.
        """
        stats_file = dataset_path / 'scene_stats.npy'
        if not stats_file.is_file():
            return None
        return np.load(stats_file, allow_pickle=True).item()

    def select_model_matrix(self, dataset_path: Path) -> np.array:
        """
        Tries to retrieve model matrix in order specified in opts.
        """
        methods = self.opt.scene_normalization.split(',')
        model_matrix = None
        for method in methods:
            method = method.lower()
            if method in ['pcd', 'camera']:
                # Try to build model matrix from scene stats.
                model_matrix = self.model_matrix_from_scene_stats(method)
            elif method == 'cache':
                model_matrix = self.load_model_matrix(dataset_path)

            if model_matrix is not None:
                self.model_matrix_mode = method
                break

        return model_matrix

    def load_model_matrix(self, dataset_path: Path) -> np.array:
        """
        Tries to load a model matrix.
        """
        model_file = dataset_path / 'model_matrix.npy'
        if not model_file.is_file():
            return None
        model = np.load(model_file, allow_pickle=True).item()
        return torch.from_numpy(model['model_matrix']).float().to(self.opt.device)

    def save_model_matrix(self, dataset_path: Path, model_matrix: torch.Tensor, overwrite=True):
        """
        Tries to load a model matrix.
        """
        model_file = dataset_path / 'model_matrix.npy'
        if overwrite or not model_file.is_file():
            np.save(model_file, {'model_matrix': model_matrix.cpu().numpy()})

    def model_matrix_from_scene_stats(self, mode=None):
        """
        Tries to build model matrix.
        """
        if self.scene_stats is None:
            return None

        radius = -1
        center = -1
        if self.scene_stats.get('pcd_radius', -1) > 0 and mode != 'camera':
            # Try from PCD stats.
            radius = self.scene_stats['pcd_radius'] * self.opt.scene_radius_scale
            center = self.scene_stats['pcd_center']
        elif self.scene_stats.get('camera_radius', -1) > 0 and mode != 'pcd':
            # Try from cameras.
            radius = self.scene_stats['camera_radius'] * self.opt.scene_radius_scale
            center = self.scene_stats['camera_intersection']

        if radius <= 0:
            # No data.
            return None

        translation = -center
        scale = self.opt.work_radius / radius

        norm_matrix = np.eye(4, dtype=np.float32)
        norm_matrix[0, 0] = scale
        norm_matrix[1, 1] = scale
        norm_matrix[2, 2] = scale
        norm_matrix[:3, 3] = translation * scale
        return torch.from_numpy(np.linalg.inv(norm_matrix)).to(self.opt.device)

    @ property
    def num_batch_on_surface_points(self) -> int:
        """
        Number of on-surface points in each batch.
        """
        if not self.opt.use_pcd:
            return 0
        return self.opt.batch_size // 2

    @ property
    def num_batch_off_surface_points(self) -> int:
        """
        Number of off-surface points in each batch.
        """
        return self.opt.batch_size - self.num_batch_on_surface_points

    def __len__(self):
        if not self.opt.use_pcd:
            # Fixed random batch count.
            return max(int(5e5 / self.opt.batch_size), 1)

        return int(np.ceil(self.coords.shape[0] / max(self.num_batch_on_surface_points, 1)))

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.num_batch_on_surface_points)

        # On surface
        on_surface_sdf = np.zeros((self.num_batch_on_surface_points, 1))  # on-surface = 0
        on_surface_coords = self.coords[rand_idcs, :]
        # Map frame index to -1,1
        on_surface_time = self.frame_indices[rand_idcs, None].astype(np.float32)
        on_surface_time = on_surface_time / max(self.num_frames - 1, 1) * 2 - 1
        if self.num_frames <= 1:
            on_surface_time[:] = common_utils.KEY_FRAME_TIME
        on_surface_normals = self.normals[rand_idcs, :]
        if self.colors is None:
            on_surface_colors = np.zeros((self.num_batch_on_surface_points, 3))
        else:
            on_surface_colors = self.colors[rand_idcs, :]

        # Off surface
        off_surface_sdf = -np.ones((self.num_batch_off_surface_points, 1))  # off-surface = -1
        off_surface_coords = np.random.uniform(-1, 1, size=(self.num_batch_off_surface_points, 3))
        off_surface_time = np.random.uniform(-1, 1, size=(self.num_batch_off_surface_points, 1))
        off_surface_normals = np.zeros((self.num_batch_off_surface_points, 3))
        off_surface_colors = np.zeros((self.num_batch_off_surface_points, 3))

        # Concatenate on+off
        sdf = np.concatenate((on_surface_sdf, off_surface_sdf), axis=0)
        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        times = np.concatenate((on_surface_time, off_surface_time), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
        colors = np.concatenate((on_surface_colors, off_surface_colors), axis=0)

        inputs = {
            'coords': torch.from_numpy(coords).float(),
            'time': torch.from_numpy(times).float(),
            'model_matrix': self.model_matrix.cpu(),
        }
        gt = {
            'sdf': torch.from_numpy(sdf).float(),
            'normals': torch.from_numpy(normals).float(),
            'colors': torch.from_numpy(colors).float(),
        }

        return inputs, gt
