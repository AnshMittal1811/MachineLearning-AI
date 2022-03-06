import os

import torch
from torch import nn

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points

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
        C,
        size,
        learn_feature=True,
        radius=1.5,
        rad_pow=2,
        accumulation_tau=1,
        accumulation='alphacomposite',
        points_per_pixel=8,
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

        self.rad_pow = rad_pow
        self.accumulation_tau = accumulation_tau
        self.accumulation = accumulation

    def forward(self, pts3D, src):
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

        # pts3D.shape = (bs, w*h, 3) --> (x,y,z) coordinate for ever element in the image raster
        # Because we have done re-projection, the i-th coordinate in the image raster must no longer be identical to (x,y)!
        # src.shape = (bs, c, w*h) --> c features for every element in the image raster (w*h)

        #print("Features: {}".format(src.shape))
        #print("3D Pointcloud: {}".format(pts3D.shape))

        # flips the x and y coordinate
        pts3D[:,:,1] = - pts3D[:,:,1]
        pts3D[:,:,0] = - pts3D[:,:,0]

        # Add on the default feature to the end of the src
        #src = torch.cat((src, self.default_feature.repeat(bs, 1, 1)), 2)

        radius = float(self.radius) / float(image_size) * 2.0 # convert radius to fit the [-1,1] NDC ?? Or is this just arbitrary scaling s.t. radius as meaningful size?
        params = compositing.CompositeParams(radius=radius)

        #print("Radius - before: {}, converted: {}".format(self.radius, radius))

        pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))
        points_idx, _, dist = rasterize_points(
            pts3D, image_size, radius, self.points_per_pixel
        ) # see method signature for meaning of these output values

        #print("points_idx: {}".format(points_idx.shape))
        #print("dist: {}".format(points_idx.shape))

        #print("Max dist: ", dist.max(), pow(radius, self.rad_pow))

        dist = dist / pow(radius, self.rad_pow) # equation 1 from the paper (3.2): this calculates N(p_i, l_mn) from the d2 dist

        #print("Max dist: ", dist.max())

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.accumulation_tau)
            .permute(0, 3, 1, 2)
        ) # equation 2 from the paper (3.2): prepares alpha values for composition of the feature vectors

        #print("alphas: ", alphas.shape)
        #print("pointclouds object: {}".format(pts3D.features_packed().shape))
        #print("alphas: ", alphas)

        if self.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0), # pts3D also contains features here, because this is now a Pointclouds object
                params,
            )
        elif self.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
                params,
            )
        elif self.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
                params,
            )
        else: raise NotImplementedError('Unsupported accumulation type: ' + self.accumulation)

        return transformed_src_alphas

if __name__ == "__main__":
    # pts3D.shape = (bs, w*h, 3) --> (x,y,z) coordinate for ever element in the image raster
    # Because we have done re-projection, the i-th coordinate in the image raster must no longer be identical to (x,y)!
    # src.shape = (bs, c, w*h) --> c features for every element in the image raster (w*h)

    bs = 1
    c = 64
    w = h = 64

    src = torch.rand((bs, c, w*h))
    print("Features: {}".format(src.shape))

    pts3D = torch.rand((bs, w*h, 3))
    print("3D Pointcloud: {}".format(pts3D.shape))

    r = RasterizePointsXYsBlending(size=32)

    reprojected_features = r(pts3D, src)
    print("Neural rendered features: {}".format(reprojected_features.shape))

