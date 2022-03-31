import torch
from abc import ABC, abstractmethod

''' Baseclass for a coordinate transform'''
# A coordinate transform a_transform = ExCoordTransform(camera_params) is then called by
# a_transform(ray_org,ray_dirs), internally executing the routine a_transform.transform()
# that must be instantiated for each individual transformation.
# /!\ Note that transforms are used in the ray sampling pipeline before the samplers!
# Thus, the form of the transforms is slightly different than in the original NeRF code.
class CoordinateTransform(ABC):
    def __init__(self, camera_params):
        self.camera_params = camera_params

    def __call__(self, ray_orgs, ray_dirs):
        return self.transform(ray_orgs, ray_dirs)

    @abstractmethod
    def transform(self, ray_orgs, ray_dirs):
        pass

''' Implementation of transforms '''
# A transform to get rays in the unitary projective cube
class ToNormalizedDeviceCoordinates(CoordinateTransform):
    def __init__(self,camera_params):
        super().__init__(camera_params)

    def transform(self, ray_orgs, ray_dirs):
        f = self.camera_params['focal']
        W = self.camera_params['W']
        H = self.camera_params['H']
        #near = self.camera_params['near']
        near = 1.0

        t = -(near + ray_orgs[:, 2]) / ray_dirs[:, 2]
        # print(f"t={t.shape}")
        ray_orgs = ray_orgs + t[:, None] * ray_dirs

        ray_orgs_x = -2. * f / H * ray_orgs[:, 0] / ray_orgs[:, 2]
        ray_orgs_y = -2. * f / W * ray_orgs[:, 1] / ray_orgs[:, 2]
        ray_orgs_z = 1. + 2. * near / ray_orgs[:, 2]

        ray_dirs_x = -2. * f / H * (ray_dirs[:, 0] / ray_dirs[:, 2] - ray_orgs[:, 0] / ray_orgs[:, 2])
        ray_dirs_y = -2. * f / W * (ray_dirs[:, 1] / ray_dirs[:, 2] - ray_orgs[:, 1] / ray_orgs[:, 2])
        ray_dirs_z = -2. * near / ray_orgs[:, 2]

        ray_orgs = torch.stack([ray_orgs_x, ray_orgs_y, ray_orgs_z], dim=1)
        ray_dirs = torch.stack([ray_dirs_x, ray_dirs_y, ray_dirs_z], dim=1)
        return ray_orgs, ray_dirs

# A transform to shift rays to a near-far system
class ToNearFar(CoordinateTransform):
    def __init__(self,camera_params):
        super().__init__(camera_params)

    def transform(self, ray_orgs, ray_dirs):
        near = self.camera_params['near']
        far = self.camera_params['far']

        ray_orgs = ray_orgs + near * ray_dirs
        ray_dirs = (far - near) * ray_dirs
        return ray_orgs, ray_dirs
