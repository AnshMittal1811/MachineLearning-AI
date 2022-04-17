import torch
import torch.nn as nn

import sys

sys.path.append("..")
import utils


class PlanarMotion(nn.Module):
    def __init__(self, n_frames, n_layers, scale=None, trans=None, **kwargs):
        """
        :param n_frames (int) N total number of frames to model
        :param n_layers (int) M number of layers
        :param scale (N, M, 2) the initial scale of each layer
        :param trans (N, M, 2) the initial translation of each layer
        """
        super().__init__()
        self.n_frames = n_frames
        self.n_layers = n_layers
        N, M = n_frames, n_layers

        if scale is None:
            scale = torch.ones(N, M, 2, dtype=torch.float32)
        if trans is None:
            trans = torch.zeros(N, M, 2, dtype=torch.float32)

        ## initialize zeros for the rotation and skew effects
        ## 8 total parameters:
        ## first four for sim transform
        ## second two extend to affine
        ## last two extend to perspective
        init = torch.zeros(N, M, 8, dtype=torch.float32)
        self.register_parameter("theta", nn.Parameter(init, requires_grad=True))
        print(f"Initialized planar motion with {N} params, {M} layers")

        self.update_scale(scale)
        self.update_trans(trans)

    def forward(self, idx, grid):
        """
        :param idx (B) which transform to evaluate
        :param grid (B, M, H, W, 3) query grid in view
        :returns cano coordinates of query grid (B, M, H, W, 2)
        """
        mat = self.theta_to_mat(self.get_theta(idx))  # (B, M, 3, 3)
        return utils.apply_homography_xy1(mat, grid)

    def get_cano2view(self, idx):
        mat = self.theta_to_mat(self.get_theta(idx))  # (B, M, 3, 3)
        return torch.linalg.inv(mat)

    def get_theta(self, idx):
        """
        :param idx (B) which transforms to index
        :returns (B, M, 8)
        """
        return self.theta[idx]

    def update_scale(self, scale):
        """
        :param scale (N, M, 2)
        """
        with torch.no_grad():
            sx = scale[..., 0:1]
            sy = scale[..., 1:2]
            s = torch.sqrt(sx * sy)
            k = torch.sqrt(sx / sy)
            self.theta[..., 0:1] = s
            self.theta[..., 4:5] = k
            print("updated scale")

    def update_trans(self, trans):
        """
        :param trans (N, M, 2)
        """
        with torch.no_grad():
            self.theta[..., 2:4] = trans
            print("updated trans")

    def theta_to_mat(self, theta):
        """
        expands the 8 parameters into 3x3 matrix
        H = [[A, t], [v.T, 1]] where A = SK + tv.T
        :param theta (N, M, 8)
        :returns mat (N, M, 3, 3)
        """
        *dims, D = theta.shape
        a = theta[..., 0:1]
        b = theta[..., 1:2]
        t = theta[..., 2:4, None]  # (*, 2, 1)
        k = theta[..., 4:5] + 1e-6
        w = theta[..., 5:6]
        vT = theta[..., None, 6:8]  # (*, 1, 2)
        SK = torch.cat([a * k, a * w + b / k, -b * k, -b * w + a / k], dim=-1).reshape(
            *dims, 2, 2
        )
        A = SK + torch.matmul(t, vT)
        return torch.cat(
            [
                torch.cat([A, t], dim=-1),
                torch.cat([vT, torch.ones_like(vT[..., :1])], dim=-1),
            ],
            dim=-2,
        )

    def get_scale(self, idx):
        """
        :param idx (B) which transforms to select
        :returns scale (B, M) of the transform
        """
        theta = self.get_theta(idx)  # (B, M, 8)
        return torch.sqrt(theta[..., 0] ** 2 + theta[..., 1] ** 2)  # (B, M)


class PlanarMotionNaive(nn.Module):
    def __init__(self, n_frames, n_layers, scales=None, trans=None, **kwargs):
        """
        :param n_frames (int) N total number of frames to model
        :param n_layers (int) M number of layers
        :param scales (N, M, 2) the initial scale of each layer
        :param trans (N, M, 2) the initial translation of each layer
        """
        super().__init__()
        self.n_frames = n_frames
        self.n_layers = n_layers
        N, M = n_frames, n_layers

        if scales is None:
            scales = torch.ones(N, M, 2, dtype=torch.float32)
        if trans is None:
            trans = torch.zeros(N, M, 2, dtype=torch.float32)

        sx = scales[..., 0:1]
        sy = scales[..., 1:2]
        tx = trans[..., 0:1]
        ty = trans[..., 1:2]
        z = torch.zeros_like(sx)

        init = torch.cat([sx, z, tx, z, sy, ty, z, z], dim=-1)
        self.register_parameter("theta", nn.Parameter(init, requires_grad=True))

    def forward(self, idx, grid):
        """
        :param idx (B) which transform to evaluate
        :param grid (B, M, H, W, 3) query grid in view
        :returns cano coordinates of query grid (B, M, H, W, 2)
        """
        mat = self.theta_to_mat(self.get_theta(idx))  # (B, M, 3, 3)
        return utils.apply_homography_xy1(mat, grid)

    def get_cano2view(self, idx):
        mat = self.theta_to_mat(self.get_theta(idx))  # (B, M, 3, 3)
        return torch.linalg.inv(mat)

    def get_theta(self, idx):
        """
        :param idx (B) which transforms to index
        :returns (B, M, 8)
        """
        return self.theta[idx]

    def theta_to_mat(self, theta):
        """
        expands the 8 parameters into 3x3 matrix
        :param theta (N, M, 8)
        :returns mat (N, M, 3, 3)
        """
        *dims, D = theta.shape
        ones = torch.ones_like(theta[..., :1])
        return torch.cat([theta, ones], dim=-1).view(*dims, 3, 3)

    def get_scale(self, idx):
        """
        :param idx (B) which transforms to select
        :returns scale (B, M) of the transform
        """
        theta = self.get_theta(idx)  # (B, M, 8)
        return 0.5 * (theta[..., 0] + theta[..., 2])  # (B, M)
