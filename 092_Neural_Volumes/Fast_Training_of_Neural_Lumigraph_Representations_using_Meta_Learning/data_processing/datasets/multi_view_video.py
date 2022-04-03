"""
Represents video of multiview frames.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import utils.math_utils as math_utils
import utils.math_utils_torch as mut
import utils.common_utils as common_utils
from data_processing.datasets.multi_view_frame import MultiViewFrame


class MultiViewVideo(Dataset):
    """
    Represents video of multiview frames.
    """

    def __init__(self, dataset_path: Path, opt):
        self.opt = opt

        # Prepare attribs.
        self.rnd = np.random.RandomState(14875)
        self.frames = []
        self.db = None

        self.colors = torch.zeros((0, 3)).float()
        self.masks = torch.zeros((0, )).float()
        self.pixels = torch.zeros((0, 2)).bool()
        self.view_ids = torch.zeros((0, )).long()
        self.frame_ids = torch.zeros((0, )).long()

        # Load frames.
        self._load_frames(dataset_path)

        # Gather all pixels.
        self.use_locals = (self.opt.load_images and
                           (not hasattr(opt, 'is_test_only') or not opt.is_test_only) and
                           (self.db is None))
        if self.use_locals and len(self.image_views) > 0:
            self.build_pixels()

    @torch.no_grad()
    def build_pixels(self):
        """
        Prepares rays for given model.
        """
        all_colors = []
        all_masks = []
        all_pixels = []
        all_view_ids = []
        all_frame_ids = []

        all_proj_matrices = []
        all_view_matrices = []
        all_resolutions = []

        print('Building rays for video...')
        for i, frame in enumerate(self.frames):
            print(f'\tAdding pixels for frame {i} of {len(self.frames)}...')
            colors, mask, pixels, view_ids, proj_matrices, view_matrices, resolutions = frame.get_all_pixels()
            all_colors += [colors]
            all_masks += [mask]
            all_pixels += [pixels]
            all_view_ids += [view_ids]
            all_frame_ids += [torch.zeros_like(view_ids).fill_(i)]

            all_proj_matrices += [proj_matrices]
            all_view_matrices += [view_matrices]
            all_resolutions += [resolutions]

        # Merge.
        self.colors = torch.cat(all_colors, 0)
        self.masks = torch.cat(all_masks, 0)
        self.pixels = torch.cat(all_pixels, 0)
        self.view_ids = torch.cat(all_view_ids, 0)
        self.frame_ids = torch.cat(all_frame_ids, 0)

        self.proj_matrices = torch.stack(all_proj_matrices, 0)
        self.view_matrices = torch.stack(all_view_matrices, 0)
        self.resolutions = torch.stack(all_resolutions, 0)

        print(f'Loaded {self.colors.shape[0]} pixels.')

    def _load_frames(self, dataset_path: Path):
        """
        Finds all frames and loads them as datasets.
        """
        if self.opt.dataset_type == 'sinesdf_static':
            # Just single frame.
            self.frames = [MultiViewFrame(dataset_path, self.opt)]
            return

        # Find explicit frames.
        frame_dirs = sorted([x for x in dataset_path.iterdir() if x.is_dir()])

        # Load full set of frames.
        for i, frame_dir in enumerate(frame_dirs):
            if self.db is not None and i in [0, len(frame_dirs) - 1]:
                self.frames += [None]

            print(f'[{i}/{len(frame_dirs)}] Loading multiview frame {frame_dir}...')
            self.frames += [MultiViewFrame(frame_dir, self.opt)]

    @property
    def image_views(self) -> list:
        """
        Shortcut to the image views of the first frame.
        """
        return self.frames[0].image_views

    @property
    def num_frames(self) -> int:
        """
        Number of video frames.
        """
        return len(self.frames)

    def parameters(self):
        """
        Gets learnable parameters.
        Camera poses and projections.
        """
        params = []
        # Only 1st frame. Others share.
        for frame in self.frames[:1]:
            params += frame.parameters()
        return params

    def _len(self):
        if self.db is not None:
            # Use DB.
            return len(self.db)

        num_rays = self.colors.shape[0]
        return int(np.ceil(num_rays / self.opt.batch_size_2d))

    def _len_ibr(self):
        # Make sure that there is only one frame, otherwise IBR model not currently supported.
        if len(self.frames) != 1:
            raise Exception('IBR model and dataset currently does not support video data.')

        if self.opt.total_number_source_views > 0:
            return self.opt.total_number_source_views

        return len(self.frames[0].image_views)

    def _getitem(self, idx):
        """
        Gets random choice of rays.
        """
        if self.db is not None:
            # Use DB.
            return self.db[idx]

        # Choose random rays.
        rand_idcs = np.random.choice(self.colors.shape[0], size=self.opt.batch_size_2d)

        colors = self.colors[rand_idcs, ...]
        mask = self.masks[rand_idcs, ...]
        ndc = self.pixels[rand_idcs, ...]
        times = self.frame_ids[rand_idcs, None].float() / (len(self.frames) - 1) * 2 - 1
        if len(self.frames) <= 1:
            times.fill_(common_utils.KEY_FRAME_TIME)
        view_ids = self.view_ids[rand_idcs, ...]

        inputs = {
            # 'rays': rays,
            'rays_ndc': ndc,  # Nx2
            'rays_time': times,  # Nx1
            'rays_view_ids': view_ids,  # N,
        }
        gt = {
            'rays_colors': colors,  # Nx3
            'rays_mask': mask,  # N,
            'rays_index': torch.from_numpy(rand_idcs),  # N, For debug only.
        }

        return inputs, gt

    def _getitem_ibr(self, idx):
        """
        Gets idx image choice of rays.
        """
        idx_mask = (self.view_ids == idx)

        colors = self.colors[idx_mask]
        mask = self.masks[idx_mask]
        # ndc = self.pixels[idx_mask]
        # times = self.frame_ids[idx_mask].unsqueeze(-1).float() / (len(self.frames) - 1) * 2 - 1
        # if len(self.frames) <= 1:
        #     times.fill_(common_utils.KEY_FRAME_TIME)
        # view_ids = self.view_ids[idx_mask]
        # frame_ids = self.frame_ids[idx_mask]
        # proj_matrices = self.proj_matrices[0, idx]
        # view_matrices = self.view_matrices[0, idx]
        # resolution = self.resolutions[0, idx]

        # Right now, only the idx number is actually used in the model.
        inputs = {
            # 'rays_ndc': ndc,  # Nx2
            # 'rays_time': times,  # Nx1
            # 'rays_view_ids': view_ids,  # N,
            # 'rays_frame_ids': frame_ids,
            # 'proj_matrix': proj_matrices.cpu(),
            # 'view_matrix': view_matrices.cpu(),
            # 'resolution': resolution
            'target_view_id': idx
        }
        gt = {
            'rays_colors': colors,  # Nx3
            'rays_mask': mask,  # N,
        }

        return inputs, gt

    def __getitem__(self, idx):
        if getattr(self.opt, 'ibr_dataset', 0):
            return self._getitem_ibr(idx)
        else:
            return self._getitem(idx)

    def __len__(self):
        if getattr(self.opt, 'ibr_dataset', 0):
            return self._len_ibr()
        else:
            return self._len()
