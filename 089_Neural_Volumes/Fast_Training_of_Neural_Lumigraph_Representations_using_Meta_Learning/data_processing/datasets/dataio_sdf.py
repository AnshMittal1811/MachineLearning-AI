from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import random

from data_processing.datasets.point_cloud_video import PointCloudVideo
from data_processing.datasets.procedural_shapes import SpherePointCloud
from data_processing.datasets.multi_view_video import MultiViewVideo
from data_processing.datasets.image_view import ImageView
import utils.math_utils as math_utils
import utils.common_utils as common_utils


class DatasetSDF(Dataset):
    """
    Joint dataset for both PCD and IMG view data.
    """

    def __init__(self, dataset_path: Path, opt):
        super().__init__()

        self.opt = opt
        self.dataset_img = None

        # PCD.
        if opt.fit_sphere:
            # Fake data.
            self.dataset_pcd = SpherePointCloud(opt)
            return

        # Real data.
        assert not opt.opt_sdf_direct
        self.dataset_pcd = PointCloudVideo(dataset_path, opt)

        # Image views.
        self.dataset_img = MultiViewVideo(dataset_path, opt)

        # Validate.
        # self.validate()

    @property
    def reference_view_index(self) -> int:
        """
        The reference image view if any.
        """
        if len(self.image_views) == 0:
            # No views.
            return -1
        if self.opt.reference_view >= 0:
            # User specified view.
            return self.opt.reference_view
        # Assume the middle is good.
        return len(self.image_views) // 2

    @property
    def reference_view(self) -> ImageView:
        """
        The reference image view if any.
        """
        index = self.reference_view_index
        if index < 0:
            return None
        return self.dataset_img.image_views[index]

    @property
    def resolution(self) -> np.array:
        """
        Resolution of the first view or default.
        """
        return self.reference_view.resolution if len(self.image_views) > 0 else np.array([512, 512], int)

    @property
    def aspect_ratio(self) -> float:
        """
        Aspect ratio of the first view.
        """
        return float(self.resolution[0]) / self.resolution[1]

    @property
    def model_matrix(self) -> torch.Tensor:
        """
        Model matrix to scale the normalized PCD to its default shape.
        """
        return self.dataset_pcd.model_matrix if self.dataset_pcd is not None else torch.from_numpy(np.eye(4, dtype=np.float32)).to(self.opt.device)

    @property
    def view_matrix(self) -> torch.Tensor:
        """
        View matrix of the first view.
        """
        if len(self.image_views) > 0:
            return self.reference_view.view_matrix
        else:
            view_matrix = torch.from_numpy(np.eye(4, dtype=np.float32)).to(self.opt.device)
            view_matrix[2, 3] = -1.8  # Camera is in Z+ and looks into Z-
            return view_matrix

    @property
    def projection_matrix(self) -> torch.Tensor:
        """
        Projection matrix of the first view.
        """
        if len(self.image_views) > 0:
            return self.reference_view.projection_matrix
        else:
            return torch.from_numpy(math_utils.getPerspectiveProjection(60, 60 / self.aspect_ratio)).to(self.opt.device)

    @property
    def coords(self) -> np.array:
        """
        Shortcut to the PCD coords.
        """
        return self.dataset_pcd.get_frame_coords(0)

    @property
    def normals(self) -> np.array:
        """
        Shortcut to the PCD normals.
        """
        return self.dataset_pcd.get_frame_normals(0)

    @property
    def colors(self) -> np.array:
        """
        Shortcut to the PCD colors.
        """
        return self.dataset_pcd.get_frame_colors(0)

    @property
    def image_views(self) -> list:
        """
        Shortcut to the image views.
        """
        return self.dataset_img.image_views if self.dataset_img is not None else []

    def frame_index_to_timestamp(self, index):
        """
        Converts frame index to [-1,1] timestamp.
        """
        if self.dataset_img is None or self.dataset_img.num_frames == 0:
            return common_utils.KEY_FRAME_TIME
        return index / max(self.dataset_img.num_frames - 1, 1) * 2 - 1

    def parameters(self):
        """
        Gets learnable parameters.
        Camera poses and projections.
        """
        if self.dataset_img is None:
            return []
        return self.dataset_img.parameters()

    def get_camera_path(self):
        """
        Returns list of view matrices for camera path flying around the object.
        """
        opt = self.opt
        if opt.video_path_type == 'linear':
            return self.get_camera_path_linear(
                -opt.video_yaw / 2, opt.video_yaw / 2,
                opt.video_num_frames,
                pivot_offset=opt.video_pivot_offset)
        elif opt.video_path_type == 'array':
            return self.get_camera_path_array(opt.video_num_frames)
        elif opt.video_path_type == 'scu':
            return self.get_camera_path_scu(opt.video_num_frames)
        elif opt.video_path_type == 'dtu':
            return self.get_camera_path_dtu(opt.video_num_frames,
                                            pivot_offset=opt.video_pivot_offset)
        else:  # eight figure
            return self.get_camera_path_interp(
                opt.video_num_frames,
                pivot_offset=opt.video_pivot_offset)

    @torch.no_grad()
    def get_camera_path_linear(self, yaw_0_deg=-90, yaw_1_deg=90, num_frames=150,
                               pivot_offset=np.array([0, 0, 0])):
        """
        Returns list of view matrices for camera path flying around the object.
        """
        # Determine the view distance from the reference camera.
        model_matrix_ref = self.model_matrix.detach().cpu().numpy()
        view_matrix_ref = self.view_matrix.detach().cpu().numpy()

        # Offset in normalized space.
        m_offset = np.eye(4, dtype=np.float32)
        m_offset[:3, 3] = pivot_offset
        model_matrix_ref = model_matrix_ref @ np.linalg.inv(m_offset)

        # Create neutral camera.
        # Camera is in Z+ and looks into Z-
        ##ref_view = np.eye(4, dtype=np.float32)
        ## camera_distance = np.linalg.norm(view_matrix_ref[:3, 3])
        ##ref_view[2, 3] = -camera_distance
        ref_view = view_matrix_ref.copy()

        # Rotate camera around y axis.
        views = []
        yaws = np.linspace(yaw_0_deg, yaw_1_deg, num_frames)
        for yaw in yaws:
            m_rot = np.eye(4, dtype=np.float32)
            m_rot[:3, :3] = Rotation.from_euler('y', yaw, degrees=True).as_matrix()
            # Apply rotation in normalized space.
            view = ref_view @ model_matrix_ref @ m_rot @ np.linalg.inv(model_matrix_ref)
            views += [torch.from_numpy(view).to(self.model_matrix.device)]
        return views

    @torch.no_grad()
    def get_camera_path_interp(self, num_frames=150,
                               pivot_offset=np.array([0, 0, 0]), angle_scale=0.5,
                               view_matrix_ref: np.array = None, view_inds=None, shape='circle'):
        """
        Returns list of view matrices for camera path
        that itnerpolates the camera pose range.
        """
        # Determine the view distance from the reference camera.
        model_matrix_ref = self.model_matrix.detach().cpu().numpy()
        if view_matrix_ref is None:
            view_matrix_ref = self.view_matrix.detach().cpu().numpy()

        # Offset in normalized space.
        m_offset = np.eye(4, dtype=np.float32)
        m_offset[:3, 3] = pivot_offset
        model_matrix_ref = model_matrix_ref @ np.linalg.inv(m_offset)

        # Select views.
        views = self.image_views
        if view_inds is not None:
            views = [views[i] for i in view_inds]

        # Determine yaw nd pitch range.
        yp_min = np.array([0, 0], np.float32)
        yp_max = np.array([0, 0], np.float32)
        for view in views:
            view_matrix = view.view_matrix.detach().cpu().numpy()
            delta_rot = (view_matrix @ model_matrix_ref) @ np.linalg.inv(view_matrix_ref @ model_matrix_ref)
            yaw_pitch = Rotation.from_matrix(delta_rot[:3, :3]).as_euler('yxz', degrees=True)[:2]
            yp_min = np.minimum(yp_min, yaw_pitch)
            yp_max = np.maximum(yp_max, yaw_pitch)
        yp_range = yp_max - yp_min
        yp_mean = (yp_min + yp_max) / 2

        # Generate 8-path.
        views = []
        ts = np.linspace(0, 2 * np.pi, num_frames)
        for t in ts:
            if shape == 'eight':
                # Figure 8
                angles = np.array([np.sin(t), 2 * np.sin(t) * np.cos(t)])
            else:
                # Circle
                angles = np.array([np.sin(t), np.cos(t)])

            angles = angles * 0.5 * angle_scale * yp_range + yp_mean
            m_rot = np.eye(4, dtype=np.float32)
            m_rot[:3, :3] = Rotation.from_euler('yx', angles, degrees=True).as_matrix()
            # Apply rotation in normalized space.
            view = view_matrix_ref @ model_matrix_ref @ m_rot @ np.linalg.inv(model_matrix_ref)
            views += [torch.from_numpy(view).to(self.model_matrix.device)]
        return views

    @torch.no_grad()
    def get_linear_interp_frames(self):
        views = []
        view_left = self.image_views[20].view_matrix.detach().cpu().numpy()
        view_right = self.image_views[19].view_matrix.detach().cpu().numpy()
        view_middle = self.image_views[21].view_matrix.detach().cpu().numpy()

        views += [torch.from_numpy(view_left).to(self.model_matrix.device)]
        for interp_amt in [0.25, 0.5, 0.75]:
            view = math_utils.interpolate_views(view_left, view_middle, interp_amt)
            views += [torch.from_numpy(view).to(self.model_matrix.device)]
        views += [torch.from_numpy(view_middle).to(self.model_matrix.device)]
        for interp_amt in [0.25, 0.5, 0.75]:
            view = math_utils.interpolate_views(view_middle, view_right, interp_amt)
            views += [torch.from_numpy(view).to(self.model_matrix.device)]
        views += [torch.from_numpy(view_right).to(self.model_matrix.device)]
        return views

    @torch.no_grad()
    def get_camera_path_array(self, num_frames=150):
        """
        Centralized figure 8 for array.
        """
        view_bottom = self.image_views[18].view_matrix.detach().cpu().numpy()
        view_top = self.image_views[21].view_matrix.detach().cpu().numpy()
        reference_view = math_utils.interpolate_views(view_bottom, view_top, 0.5)
        return self.get_camera_path_interp(num_frames=num_frames,
                                           pivot_offset=np.array([0, 0, 0]), angle_scale=[0.5, 4.0],
                                           view_matrix_ref=reference_view, view_inds=np.arange(16, 22))

    @torch.no_grad()
    def get_camera_path_scu(self, num_frames=150):
        """
        Swipe the central cameras back and forth.
        """
        #views = [x.view_matrix.detach().cpu().numpy() for x in self.image_views[:-1]]

        view_left = self.image_views[1].view_matrix.detach().cpu().numpy()
        view_right = self.image_views[4].view_matrix.detach().cpu().numpy()
        reference_view = math_utils.interpolate_views(view_left, view_right, 0.5)
        return self.get_camera_path_interp(num_frames=num_frames,
                                           pivot_offset=np.array([0, 0, 0]), angle_scale=0.5,
                                           view_matrix_ref=reference_view, view_inds=np.arange(6))

    @torch.no_grad()
    def get_camera_path_dtu(self, num_frames, pivot_offset):
        return self.get_camera_path_interp(num_frames=num_frames,
                                           pivot_offset=np.array(pivot_offset), angle_scale=0.25)

    def validate(self):
        """
        Computes projection errors.
        """
        import data_processing.components.conversions as conversions
        import open3d as o3d
        import imageio
        import matplotlib.pyplot as plt

        # Rebuild pcd.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.dataset_pcd.coords)
        pcd.normals = o3d.utility.Vector3dVector(self.dataset_pcd.normals)
        pcd.colors = o3d.utility.Vector3dVector(self.dataset_pcd.colors * 0.5 + 0.5)

        for i, view in enumerate(self.dataset_img.image_views):
            # Inputs
            image_gt = view.image.permute(1, 2, 0).cpu().numpy()
            resolution = np.array([image_gt.shape[1], image_gt.shape[0]], int)

            # Render PCD.
            render, render_coords, render_colors = conversions.render_pcd(
                pcd,
                (view.view_matrix @ self.model_matrix).detach().cpu().numpy(),
                view.projection_matrix.cpu().detach().numpy(), resolution)

            # _, axs = plt.subplots(1, 2)
            # axs[0].imshow(image_gt)
            # axs[1].imshow(render)
            # plt.show()
            out_path = Path(self.opt.logging_root) / self.opt.experiment_name / 'validation'
            out_path.mkdir(0o777, True, True)
            conversions.imwritef(out_path / f'view_{i:03d}_pcd.png', render)
            conversions.imwritef(out_path / f'view_{i:03d}_gt.png', image_gt)

            # Compute error.
            gt = image_gt[render_coords[:, 1], render_coords[:, 0], :]
            error = ((render_colors - gt) ** 2).mean()
            print(f'Projection to View #{i} => Error = {error:.3f}')

    def __len__(self):
        if self.dataset_img is not None and len(self.dataset_img) > 1:
            return len(self.dataset_img)
        else:
            return len(self.dataset_pcd)

    def __getitem__(self, idx):
        if not getattr(self.opt, "is_test_only", 0):
            if idx not in self.opt.TRAIN_VIEWS:
                idx = random.choice(self.opt.TRAIN_VIEWS)

        # Sample PCD.
        inputs, gt = self.dataset_pcd[idx]

        # Add image rays.
        if (self.opt.opt_render_shape or getattr(self.opt, 'opt_render_color', 0) or getattr(self.opt, 'ibr_dataset', 0)) and len(self.image_views) > 0:
            im_inputs, im_gt = self.dataset_img[idx]
            inputs.update(im_inputs)
            gt.update(im_gt)

        return inputs, gt
