
import torch
import torch.nn as nn

from pytorch3d.structures import Pointclouds

EPS = 1e-2


def get_pixel_grids(height, width):
    with torch.no_grad():
        # texture coordinate
        x_linspace = torch.linspace(0, width - 1, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0, height - 1, height).view(height, 1).expand(height, width)
        x_coordinates = x_linspace.contiguous().view(-1)
        y_coordinates = y_linspace.contiguous().view(-1)
        ones = torch.ones(height * width)
        indices_grid = torch.stack([x_coordinates,  y_coordinates, ones, torch.ones(height * width)], dim=0)
    return indices_grid

def get_splatter(
    name, depth_values, opt=None, size=(150, 200), C=64, points_per_pixel=8
):
    if name == "xyblending":
        from models.layers.z_buffer_layers import RasterizePointsXYsBlending

        return RasterizePointsXYsBlending(
            C,
            learn_feature=opt.learn_default_feature,
            radius=opt.radius,
            size=size,
            points_per_pixel=points_per_pixel,
            opts=opt,
        )
    elif name == "pulsar":
        ball_num = size[1] * size[0]
        from models.layers.z_buffer_layers import Ortho_PulsarRender
        return  Ortho_PulsarRender(C, width=size[1], height=size[0], radius=opt.radius, max_num_balls=ball_num, orthogonal_projection=True, learn_feature=opt.learn_default_feature,  opt=opt)
    else:
        raise NotImplementedError()

def get_ptsmanipulator(name):
    
    if name == "Screen":
        return Screen_PtsManipulator

    elif name == "NDC":
        return NDC_PtsManipulator
    
    else: 
        raise NotImplementedError


class NDC_PtsManipulator(nn.Module):
    def __init__(self, W, H=None, C=64, opt=None):
        super().__init__()
        self.opt = opt

        self.splatter = get_splatter(
            opt.splatter, None, opt, size=(H,W), C=C, points_per_pixel=opt.pp_pixel
        )
        if H is None:
            H = W
        self.scale_factor = opt.scale_factor

        xs = torch.linspace(0.5, W - 0.5, W) / float(W - 1) * 2 - 1
        ys = torch.linspace(0.5, H - 0.5, H) / float(H - 1) * 2 - 1

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat(
            (xs, -ys, -torch.ones(xs.size()), torch.ones(xs.size())), 1
        ).view(1, 4, -1)

        self.register_buffer("xyzs", xyzs)

    def view_to_world_coord(self, pts3D, K, K_inv, RT_cam1, RTinv_cam1):
        # PERFORM PROJECTION
        # Project the world points into the new view
        bs = pts3D.shape[0]
        if len(pts3D.size()) > 3:
        # reshape into the right positioning
            pts3D = pts3D.contiguous().view(bs, 1, -1)
        projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1
        cam1_X = K_inv.bmm(projected_coors)


        # Transform into world coordinates
        wrld_X = RTinv_cam1.bmm(cam1_X)
        return wrld_X

    def world_to_view(self, pts3D, K, K_inv, RT_cam2, RTinv_cam2):

        wrld_X = RT_cam2.bmm(pts3D)
        xy_proj = K.bmm(wrld_X)

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)
        sampler[mask.repeat(1, 3, 1)] = -10
        # Flip the ys
        # sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(
        #     2
        # ).to(sampler.device)

        return sampler


class Screen_PtsManipulator(nn.Module):
    def __init__(self, W, H=None, C=64, opt=None):
        super().__init__()
        self.opt = opt

        self.splatter = get_splatter(
            opt.splatter, None, opt, size=(H,W), C=C, points_per_pixel=opt.pp_pixel
        )
        if H is None:
            H = W
            
        self.H = H 
        self.W = W
        self.scale_factor = opt.scale_factor
        
        xyzs = get_pixel_grids(height=H, width=W).view(1,  4, -1)

        self.register_buffer("xyzs", xyzs)

    def view_to_world_coord(self, pts3D, K, K_inv, RT_cam1, RTinv_cam1):
        # PERFORM PROJECTION
        # Project the world points into the new view
        bs = pts3D.shape[0]
        if len(pts3D.size()) > 3:
        # reshape into the right positioning
            pts3D = pts3D.contiguous().view(bs, 1, -1)
        projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1
        cam1_X = K_inv.bmm(projected_coors)


        # Transform into world coordinates
        wrld_X = RTinv_cam1.bmm(cam1_X)
        return wrld_X

    def world_to_view_screen(self, pts3D, K, K_inv, RT_cam2, RTinv_cam2):

        wrld_X = RT_cam2.bmm(pts3D)
        xy_proj = K.bmm(wrld_X)
        
        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS
        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)
        
        # Remove invalid zs that cause nans
        sampler[mask.repeat(1, 3, 1)] = -10
        # sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(2).to(sampler.device)
        return sampler
        
    def world_to_view(self, pts3D, K, K_inv, RT_cam2, RTinv_cam2):

        wrld_X = RT_cam2.bmm(pts3D)
        xy_proj = K.bmm(wrld_X)
        
        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS
        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)
        
        # Remove invalid zs that cause nans
        sampler[mask.repeat(1, 3, 1)] = -10
        
        # transfer to NDC coordinate
        sampler[:,0] = (sampler[:, 0] / (self.W -1) * 2.0 -1.0)
        sampler[:,1] = (sampler[:, 1] / (self.H -1) * 2.0 -1.0)
        # Flip the ys
        # sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(2).to(sampler.device)

        return sampler
  
