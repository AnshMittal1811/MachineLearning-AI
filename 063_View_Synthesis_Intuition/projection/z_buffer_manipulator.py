import torch
import torch.nn as nn

EPS = 1e-2

def get_splatter(
        name, size, C, points_per_pixel, learn_feature, radius, rad_pow, accumulation, accumulation_tau
):
    if name == "xyblending":
        from projection.z_buffer_layers import RasterizePointsXYsBlending

        return RasterizePointsXYsBlending(
            C=C,
            learn_feature=learn_feature,
            radius=radius,
            rad_pow=rad_pow,
            size=size,
            points_per_pixel=points_per_pixel,
            accumulation=accumulation,
            accumulation_tau=accumulation_tau
        )

    else:
        raise NotImplementedError()

class PtsManipulator(nn.Module):

    matterport_mode = 'mp3d'
    icl_nuim_mode = 'icl'
    modes = [matterport_mode, icl_nuim_mode]

    def __init__(self,
                 mode='mp3d',
                 W=640,
                 H=480,
                 C=3,
                 learn_feature=True,
                 radius=1.5,
                 points_per_pixel=8,
                 accumulation_tau=1,
                 rad_pow=2,
                 accumulation='alphacomposite'
                 ):
        super().__init__()

        self.mode = mode
        if mode not in PtsManipulator.modes:
            raise ValueError("Unsupported mode: " + mode)

        self.splatter = get_splatter(
            name="xyblending",
            size=W,
            C=C,
            points_per_pixel=points_per_pixel,
            learn_feature=learn_feature,
            radius=radius,
            rad_pow=rad_pow,
            accumulation=accumulation,
            accumulation_tau=accumulation_tau
        )

        if mode == PtsManipulator.icl_nuim_mode:
            self.img_shape = (H, W)

            # create coordinate system for x and y
            xs = torch.linspace(0, W - 1, W)
            ys = torch.linspace(0, H - 1, H)

            xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
            ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

            # build homogeneous coordinate system with [X, Y, 1, 1] to prepare for depth
            xyzs = torch.cat(
                (xs, ys, torch.ones(xs.size()), torch.ones(xs.size())), 1
            ).view(1, 4, -1)

        elif mode == PtsManipulator.matterport_mode:
            xs = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1
            ys = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1

            xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
            ys = ys.view(1, 1, W, 1).repeat(1, 1, 1, W)

            xyzs = torch.cat(
                (xs, -ys, -torch.ones(xs.size()), torch.ones(xs.size())), 1
            ).view(1, 4, -1)

        self.register_buffer("xyzs", xyzs)

    def project_pts(
            self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, dynamics
    ):
        # add Zs to the coordinate system
        # projected_coors is then [X*Z, -Y*Z, -Z, 1] with Z being the depth of the image
        projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(projected_coors)

        # Transform to World Coordinates with RT of input view
        wrld_X = RT_cam1.bmm(cam1_X)

        # Add dynamic changes if available
        if dynamics:
            bs = wrld_X.shape[0] # batch size
            transformation = dynamics["transformation"] # retrieve dynamic transformation from data
            mask = dynamics["input_mask"].view(bs, -1) # retrieve mask from data and flatten because wrld_X is flattened, too.
            #wrld_X[:, :, mask] = transformation.bmm(wrld_X[:, :, mask])
            # TODO how to vectorize this?
            for i in range(bs):
                wrld_X[i, :, mask[i]] = transformation[i].matmul(wrld_X[i, :, mask[i]]) # apply transformation to all masked points in the point cloud

        # Transform from World coordinates to camera of output view
        new_coors = RTinv_cam2.bmm(wrld_X)

        # Apply intrinsics / go back to image plane
        xy_proj = K.bmm(new_coors)

        # remove invalid zs that cause nans
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        if self.mode == PtsManipulator.icl_nuim_mode:
            # here we concatenate (x,y) / z and the original z-coordinate into a new (x,y,z) vector
            sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)

            # rescale coordinates to work with splatting and move to origin
            sampler[:, 0, :] = sampler[:, 0, :] / float(self.img_shape[1] - 1) * 2 - 1
            sampler[:, 1, :] = sampler[:, 1, :] / float(self.img_shape[0] - 1) * 2 - 1

            # here we set (x,y,z) to -10 where we have invalid zs that cause nans
            sampler[mask.repeat(1, 3, 1)] = 10

        elif self.mode == PtsManipulator.matterport_mode:
            sampler = torch.cat((xy_proj[:, 0:2, :] / -zs, xy_proj[:, 2:3, :]), 1)
            sampler[mask.repeat(1, 3, 1)] = -10
            # Flip the ys
            sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(
                2
            ).to(sampler.device)

        return sampler

    def forward_justpts(
            self, src, pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, dynamics=None
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, dynamics
        )
        pointcloud = pts3D.permute(0, 2, 1).contiguous()
        result = self.splatter(pointcloud, src)

        return result

    def forward(
            self,
            alphas,
            src,
            pred_pts,
            K,
            K_inv,
            RT_cam1,
            RTinv_cam1,
            RT_cam2,
            RTinv_cam2,
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)
            alphas = alphas.view(bs, 1, -1).permute(0, 2, 1).contiguous()

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        result = self.splatter(pts3D.permute(0, 2, 1).contiguous(), alphas, src)

        return result