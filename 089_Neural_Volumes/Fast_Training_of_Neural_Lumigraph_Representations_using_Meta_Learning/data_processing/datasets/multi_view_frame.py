"""
Represents single multiview 2D image collection.
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
from data_processing.datasets.image_view import ImageView


class MultiViewFrame(object):
    """
    Represents a set of views on one scene.
    """

    def __init__(self, dataset_path: Path, opt):
        self.opt = opt

        # Load image views.
        self.rnd = np.random.RandomState(14875)
        self.image_views = []
        self._load_image_views(dataset_path)

        # Filter train/test
        self.test_view_ids = common_utils.parse_comma_int_args(opt.test_views)

    def get_all_pixels(self) -> np.array:
        """
        Prepares rays for given model.
        """
        all_colors = []
        all_masks = []
        all_pixels = []
        all_view_ids = []

        all_projection_matrices = []
        all_view_matrices = []
        all_resolutions = []

        # Append all views' rays.
        for i, view in enumerate(self.image_views):
            # Skip test views! This is for training only.
            if i in self.test_view_ids:
                continue

            # Get colors.
            image = view.image
            mask = view.mask
            mask = mask > 0

            # Mask colors.
            colors = image.permute(1, 2, 0).reshape(-1, image.shape[0])
            mask = mask.reshape(-1)

            # Query the pixel NDCs.
            pixels = view.get_pixels_ndc('cpu')
            view_ids = torch.zeros((pixels.shape[0],)).long().fill_(i)

            # Gather.
            all_colors += [colors]
            all_masks += [mask]
            all_pixels += [pixels]
            all_view_ids += [view_ids]

            all_projection_matrices += [view.projection_matrix]
            all_view_matrices += [view.view_matrix]
            all_resolutions += [torch.from_numpy(view.resolution)]

        # Concatenate vertically.
        all_colors = torch.cat(all_colors, 0)
        all_masks = torch.cat(all_masks, 0)
        all_pixels = torch.cat(all_pixels, 0)
        all_view_ids = torch.cat(all_view_ids, 0)

        all_projection_matrices = torch.stack(all_projection_matrices, 0)
        all_view_matrices = torch.stack(all_view_matrices, 0)
        all_resolutions = torch.stack(all_resolutions, 0)

        return all_colors, all_masks, all_pixels, all_view_ids, \
            all_projection_matrices, all_view_matrices, all_resolutions

    def _load_image_views(self, dataset_path: Path):
        """
        Finds all images and loads them as datasets.
        """
        img_files = sorted([x for x in dataset_path.iterdir() if x.suffix in ['.png', '.jpg', '.jpeg']])
        img_files = sorted([x for x in img_files if x.stem.endswith('_rgb')])
        print(f'\tLoading {len(img_files)} image views...')
        for im_file in img_files:
            # Must have metadata.
            meta_file = im_file.parent / (im_file.stem + '_meta.npy')
            if not meta_file.is_file():
                continue
            #print(f'\tLoading Image view from {im_file}.')
            self.image_views += [ImageView(im_file, self.opt)]

        if self.opt.randomize_cameras:
            print('Randomizing camera poses...')
            cam_poses = np.array([view.meta_view_matrix[:3, 3] for view in self.image_views])
            sigma = max(np.std(cam_poses, 0).max() / np.sqrt(len(self.image_views)) * 0.5, 0.02)
            for view in self.image_views:
                view_matrix = view.meta_view_matrix.copy()
                view_matrix[:3, 3] += self.rnd.normal(0, sigma, 3)
                view.setup_camera_params(view_matrix, view.meta_projection_matrix)

    def parameters(self):
        """
        Gets learnable parameters.
        Camera poses and projections.
        """
        params = []
        for i, view in enumerate(self.image_views):
            if i in self.test_view_ids:
                continue
        return params
