"""
Represents single 2D image.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torchgeometry as tgm

import utils.math_utils as math_utils
import utils.math_utils_torch as mut
import utils.common_utils as common_utils
import sdf_rendering


class ImageView(object):
    """
    Represents RGB image taken from a certain viewpoint.
    """

    def __init__(self, image_filename: Path, opt):
        super().__init__()

        self.opt = opt
        self.name = image_filename.stem
        self.image_filename = image_filename

        # Requires metadata.
        meta_file = image_filename.parent / (image_filename.stem + '_meta.npy')
        meta = np.load(meta_file, allow_pickle=True)
        self.meta_view_matrix = meta.item()['view'].astype(np.float32)
        self.meta_projection_matrix = meta.item()['projection'].astype(np.float32)
        self.meta_resolution = meta.item()['resolution'].astype(int)

        # Load image.
        self.image = None
        self.mask = None
        if self.opt.load_images:
            self.load_image(image_filename)

            if opt.load_im_scale < 1.0:
                # Hard-downscale.
                self.meta_resolution = np.array(self.meta_resolution.astype(float) *
                                                opt.load_im_scale + 0.5).astype(int)
                self.image = self.get_image_resized(self.meta_resolution)
                self.mask = self.get_mask_resized(self.meta_resolution)

        # Prepare camera parameters for optimization.
        self.setup_camera_params(self.meta_view_matrix, self.meta_projection_matrix)

    def setup_camera_params(self, view_matrix: np.array, projection_matrix: np.array):
        """
        Creates optimizeable parameters for camera pose learning.
        """
        # Decompose projection.
        proj_params = math_utils.decompose_projection_matrix(projection_matrix)
        self.near = torch.tensor([proj_params['n']], dtype=torch.float).to(self.opt.device)
        self.far = torch.tensor([proj_params['f']], dtype=torch.float).to(self.opt.device)
        btlr = torch.from_numpy(np.array([proj_params['b'], proj_params['t'],
                                          proj_params['l'], proj_params['r']], np.float32))

        # View matrix. Quaternion and tvec = 4 + 3
        r_quat = torch.from_numpy(R.from_matrix(view_matrix[:3, :3]).as_quat()).float()
        t_vec = torch.from_numpy(view_matrix[:3, 3])

        # Optimizeable params.
        self.intrinsics = nn.Parameter(btlr.to(self.opt.device).requires_grad_(True))
        self.extrinsics = nn.Parameter(torch.cat((r_quat, t_vec)).to(self.opt.device).requires_grad_(True))

    @ property
    def projection_matrix(self) -> torch.Tensor:
        """
        Rebuilds differentiable projection matrix.
        """
        return mut.glFrustrum(self.intrinsics[0:1],
                              self.intrinsics[1:2],
                              self.intrinsics[2:3],
                              self.intrinsics[3:4],
                              self.near,
                              self.far)[0]

    @ property
    def view_matrix(self) -> torch.Tensor:
        """
        Rebuilds differentiable view matrix.
        """
        r_quat = self.extrinsics[0:4]
        t_vec = self.extrinsics[4:7]
        r_mat = mut.quaternion_to_matrix(r_quat)[0]
        view_3x4 = torch.cat((r_mat, t_vec[:, None]), 1)
        padding = torch.tensor([[0, 0, 0, 1]], dtype=torch.float).to(self.extrinsics.device)
        return torch.cat((view_3x4, padding), 0)

    def renormalize_poses(self):
        """
        Noramlizes camera quaternions.
        """
        with torch.no_grad():
            self.extrinsics[0:4].div_(torch.norm(self.extrinsics[0:4]))

    def load_image(self, image_filename: Path):
        """
        Loads an image file and precomputes the scales.
        """
        image = Image.open(image_filename)

        # To tensor.
        transform = Compose([
            # Resize(sidelength),
            ToTensor(),
            # Normalize(torch.tensor([0.5]), torch.tensor([0.5]))
        ])
        image = transform(image)

        # Has mask?
        mask_file = image_filename.parent / (image_filename.stem[:-4] + '_mask.png')
        if mask_file.is_file():
            mask = Image.open(mask_file)
            mask = transform(mask)[:1, ...]
        else:
            mask = torch.ones_like(image)[:1, ...]

        self.image = image
        self.mask = mask

    def get_image_resized(self, resolution) -> torch.Tensor:
        """
        Gets the image resized.
        """
        if self.image is None:
            return None
        if self.image.shape[2] == resolution[0] and self.image.shape[1] == resolution[1]:
            return self.image
        return F.interpolate(self.image[None, ...], tuple(resolution[::-1]), mode='area')[0, ...]

    def get_mask_resized(self, resolution) -> torch.Tensor:
        """
        Gets the image resized.
        """
        if self.mask is None:
            return None
        if self.mask.shape[2] == resolution[0] and self.mask.shape[1] == resolution[1]:
            return self.mask
        return F.interpolate(self.mask[None, ...], tuple(resolution[::-1]), mode='area')[0, ...]

    @ property
    def resolution(self) -> np.array:
        """
        The default resolution.
        """
        return self.meta_resolution

    def get_rays(self, model_matrix: torch.Tensor):
        """
        Gets view rays for given model.
        """
        # Build rays.
        rays_o, rays_d = sdf_rendering.get_rays_all(
            torch.from_numpy(self.resolution).to(self.opt.device),
            model_matrix,
            self.view_matrix,
            self.projection_matrix)
        return rays_o, rays_d

    def get_pixels_ndc(self, device=None):
        """
        Gets view NDC pixels for given model.
        """
        resolution = torch.from_numpy(self.resolution)
        if device is not None:
            resolution = resolution.to(device)
        return sdf_rendering.get_pixels_ndc(resolution)

    @staticmethod
    def crop_image(image: torch.Tensor, projection_matrix: np.array,
                   crop_bbox: np.array):
        """
        Crops image and its projecton matrix.
        """
        # Random offset.
        crop_bbox = np.array(crop_bbox, int)
        start_idx = crop_bbox[:2]
        size = crop_bbox[2:]
        end_idx = start_idx + size

        # Crop the image.
        im_crop = image[:, start_idx[1]:end_idx[1], start_idx[0]:end_idx[0]]

        # Relative viewport.
        im_size = np.array([image.shape[2], image.shape[1]], int)
        crop_start = start_idx / (im_size.astype(np.float32) - 1)
        crop_size = (size - 1) / (im_size.astype(np.float32) - 1)
        crop_bb = np.concatenate((crop_start, crop_size))

        # The 3D coordinates have Y (down->up) vs. 2D: Y (up->down)
        crop_bb[1] = 1 - crop_bb[1] - crop_bb[3]

        # Compute appropriate projection matrix.
        m_proj_crop = math_utils.crop_projection_matrix(projection_matrix, crop_bb)

        return im_crop, m_proj_crop
