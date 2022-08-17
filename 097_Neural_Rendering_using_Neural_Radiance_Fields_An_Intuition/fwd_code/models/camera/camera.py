import math
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import Rotate, Transform3d, Translate

from pytorch3d.renderer.cameras import CamerasBase

_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)

class IdentifyCamera(CamerasBase):
    """
    A camera base which is basically identify function.
    """

    def __init__(
        self,
        focal_length=1.0,
        principal_point=((0.0, 0.0),),
        R=_R,
        T=_T,
        K=None,
        device="cpu",
        image_size=((-1, -1),),
    ):

        super().__init__(
        device=device,
        focal_length=focal_length,
        principal_point=principal_point,
        R=R,
        T=T,
        K=K,
        image_size=image_size,
        )
        
    def get_projection_transform(self, **kwargs) -> Transform3d:

        K = torch.zeros((self.N, 4, 4), dtype=torch.float32 ).to(self.device) 

        K[:, 0, 0] = 1.0
        K[:, 1, 1] = 1.0
        K[:, 2, 2] = 1.0
        K[:, 3, 3] = 1.0

        transform = Transform3d(device=self.device)
        transform._matrix = K.transpose(1, 2).contiguous()
        return transform
    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs
        ) -> torch.Tensor:
        """>!
        FoV cameras further allow for passing depth in world units
        (`scaled_depth_input=False`) or in the [0, 1]-normalized units
        (`scaled_depth_input=True`)

        Args:
            scaled_depth_input: If `True`, assumes the input depth is in
                the [0, 1]-normalized units. If `False` the input depth is in
                the world units.
        """

        # obtain the relevant transformation to ndc
        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform()
        else:
            to_ndc_transform = self.get_projection_transform()

        if scaled_depth_input:
            # the input is scaled depth, so we don't have to do anything
            xy_sdepth = xy_depth
        else:
            # parse out important values from the projection matrix
            K_matrix = self.get_projection_transform(**kwargs.copy()).get_matrix()
            # parse out f1, f2 from K_matrix
            unsqueeze_shape = [1] * xy_depth.dim()
            unsqueeze_shape[0] = K_matrix.shape[0]
            f1 = K_matrix[:, 2, 2].reshape(unsqueeze_shape)
            f2 = K_matrix[:, 3, 2].reshape(unsqueeze_shape)
            # get the scaled depth
            sdepth = (f1 * xy_depth[..., 2:3] + f2) / xy_depth[..., 2:3]
            # concatenate xy + scaled depth
            xy_sdepth = torch.cat((xy_depth[..., 0:2], sdepth), dim=-1)

        # unproject with inverse of the projection
        unprojection_transform = to_ndc_transform.inverse()
        return unprojection_transform.transform_points(xy_sdepth)


    def is_perspective(self):
        return True



