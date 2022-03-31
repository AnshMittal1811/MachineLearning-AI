"""
On-the fly differentiable rays.
"""
import torch
from torch import nn
import sdf_rendering


class RayBuilder(nn.Module):
    '''Builds rays on-the-fly from NDC and view indices.'''

    def __init__(self, opt, img_dataset, model_matrix: torch.Tensor):
        super().__init__()
        self.opt = opt
        self.img_dataset = img_dataset
        self.register_buffer('model_matrix', model_matrix)

        # Register parameters for save and load support.
        params = self.img_dataset.parameters()
        for i, param in enumerate(params):
            self.register_parameter(f'cam_{i:03d}', param)

    def renormalize(self):
        """
        Ensures the quaternions remain normalized.
        """
        for view in self.img_dataset.image_views:
            view.renormalize_poses()

    def forward(self, ndc, view_ids):
        """
        Produces 8D ray vector: [ray_o, ray_d, t_min, t_max]
        """
        # Build projection and view matrices
        views = self.img_dataset.image_views
        view_projs = torch.stack([view.projection_matrix for view in views], 0)
        view_views = torch.stack([view.view_matrix for view in views], 0)

        # Sample based on view_ids
        px_projs = view_projs[view_ids, ...]
        px_views = view_views[view_ids, ...]

        # Compute rays.
        rays_o, rays_d = sdf_rendering.get_rays(ndc, self.model_matrix, px_views, px_projs)
        t_min, t_max = sdf_rendering.get_ray_limits_sphere(rays_o, rays_d)
        rays = torch.cat((rays_o, rays_d, t_min, t_max), -1)
        return rays
