import os

import torch
from torch import nn

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points
from pytorch3d.renderer.points.pulsar import Renderer as pulsarRender


torch.manual_seed(42)

class RasterizePointsXYsBlending(nn.Module):
    """
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options

    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
        self,
        C=64,
        learn_feature=True,
        radius=1.5,
        size=256,
        points_per_pixel=8,
        opts=None,
    ):
        super().__init__()
        if learn_feature:
            default_feature = nn.Parameter(torch.randn(1, C, 1))
            self.register_parameter("default_feature", default_feature)
        else:
            default_feature = torch.zeros(1, C, 1)
            self.register_buffer("default_feature", default_feature)

        self.radius = radius
        self.size = size
        self.points_per_pixel = points_per_pixel
        self.opts = opts

    def forward(self, pts3D, src, opacity=None, depth=False):
        if isinstance(pts3D,list):
            # if the pts3d has different point number for each point cloud
            bs = len(src)
            image_size = self.size 
            for i in range(len(pts3D)):
                pts3D[i][:, 1] = - pts3D[i][:, 1]
                pts3D[i][:, 0] = - pts3D[i][:, 0] 

            if len(image_size) > 1:
                radius = float(self.radius) / float(image_size[0]) * 2.0
            else:
                radius = float(self.radius) / float(image_size) * 2.0

            src = [src[i].permute(1,0) for i in range(len(src))]
            pts3D = Pointclouds(points=pts3D, features=src)

        else:
            bs = src.size(0)
            if len(src.size()) > 3:
                bs, c, w, _ = src.size()
                image_size = w

                pts3D = pts3D.permute(0, 2, 1)
                src = src.unsqueeze(2).repeat(1, 1, w, 1, 1).view(bs, c, -1)
            else:
                bs = src.size(0)
                image_size = self.size

            # Make sure these have been arranged in the same way
            assert pts3D.size(2) == 3
            assert pts3D.size(1) == src.size(2)  

            pts3D[:,:,1] = - pts3D[:,:,1]
            pts3D[:,:,0] = - pts3D[:,:,0]

            # Add on the default feature to the end of the src
            # src = torch.cat((src, self.default_feature.repeat(bs, 1, 1)), 2)

            if len(image_size) > 1:
                radius = float(self.radius) / float(image_size[0]) * 2.0
            else:
                radius = float(self.radius) / float(image_size) * 2.0

            pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))
        points_idx, z_buf, dist = rasterize_points(
            pts3D, image_size, radius, self.points_per_pixel
        )

        # if os.environ["DEBUG"]:
        #     print("Max dist: ", dist.max(), pow(radius, self.opts.rad_pow))

        dist = dist / pow(radius, self.opts.rad_pow)

        # if os.environ["DEBUG"]:
        #     print("Max dist: ", dist.max())

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.opts.tau)
            .permute(0, 3, 1, 2)
        )

        if self.opts.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.opts.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.opts.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        if depth is False:
            return transformed_src_alphas
        else:
            w_normed = alphas * (points_idx.permute(0,3,1,2) >= 0).float()
            w_normed = w_normed / w_normed.sum(dim=1, keepdim=True).clamp(min=1e-9)
            z_weighted = z_buf.permute(0,3,1,2).contiguous() * w_normed.contiguous()
            z_weighted= z_weighted.sum(dim=1, keepdim=True)
            return transformed_src_alphas, z_weighted


class Ortho_PulsarRender(nn.Module):

    def __init__(self, C, width, height, radius, max_num_balls=1e6, orthogonal_projection=True, learn_feature=True,  opt=None):

        super().__init__()
        if learn_feature:
            bg_col = nn.Parameter(torch.zeros(C))
            self.register_parameter("bg_col", bg_col)
        else:
            bg_col = torch.zeros(C)
            self.register_buffer("bg_col", bg_col)
        
        self.radius = float(radius) / float(width) * 2.0

        self.render = pulsarRender(n_channels=C, width=width, height=height, max_num_balls=int(max_num_balls), orthogonal_projection=orthogonal_projection)

        # identical camera params
        cam_rot = torch.zeros(6)
        cam_rot[0] = 1.0
        cam_rot[4] = 1.0

        base_cam_params = torch.cat((
            torch.zeros(3),
            cam_rot,
            torch.tensor([0, 2, 0, 0]))
        )

        self.register_buffer("base_cam_params", base_cam_params)
        self.opt = opt
        self.gamma = opt.gamma
        self.C =  C
        self.max_depth = opt.max_z
        self.min_depth = opt.min_z
        self.H = height
        self.W = width
        self.max_num_balls = max_num_balls
        self.orthogonal_projection = orthogonal_projection
        
    def forward(self, pts3D, src, opacity=None):
        """
        Args:
          -- pts3D: shape B, N ,3
          -- src: shape B, K, N
          -- opacity: shape B, N
        """
        B, K, N = src.shape
        assert B == pts3D.shape[0]
        assert N == src.shape[-1]

        # render = pulsarRender(n_channels=self.C, width=self.W, height=self.H, max_num_balls=int(self.max_num_balls), orthogonal_projection=self.orthogonal_projection, device=float(pts3D.get_device())).to(pts3D.device)

        vert_pos = pts3D
        vert_pos[:, :, 1] = -1 * vert_pos[:, :, 1]
        vert_pos[:, :, 1] = vert_pos[:, :, 1]  * (self.H - 1) / (self.W - 1)
        vert_col = src.permute(0, 2, 1)
        vert_rad = torch.ones((B, N), device=pts3D.device) * self.radius

        cam_params = self.base_cam_params.repeat(B, 1).clone().to(pts3D.device)
        #self.render.device_tracker = (torch.ones(1) * float(pts3D.get_device())).to(pts3D.device)
        #print("vert pos device", vert_pos.device)
        #render = self.render.to(pts3D.device)
        image = self.render(vert_pos=vert_pos, vert_col=vert_col, vert_rad=vert_rad, bg_col=self.bg_col, cam_params=cam_params, gamma=self.gamma, max_depth=self.max_depth, min_depth=self.min_depth,opacity=opacity).permute(0, 3, 1, 2)
        # del render
        return image
        # return None
