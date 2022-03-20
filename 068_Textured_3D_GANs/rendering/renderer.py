import kaolin
if kaolin.__version__ == '0.1.0':
    from kaolin.graphics.dib_renderer.rasterizer import linear_rasterizer
    from kaolin.graphics.dib_renderer.utils import datanormalize

from .fragment_shader import fragmentshader

import torch
import torch.nn as nn

def ortho_projection(points_bxpx3, faces_fx3):
    xy_bxpx3 = points_bxpx3 # xyz
    xy_bxpx2 = xy_bxpx3[:, :, :2] # xy

    pf0_bxfx3 = points_bxpx3[:, faces_fx3[:, 0], :]
    pf1_bxfx3 = points_bxpx3[:, faces_fx3[:, 1], :]
    pf2_bxfx3 = points_bxpx3[:, faces_fx3[:, 2], :]
    points3d_bxfx9 = torch.cat((pf0_bxfx3, pf1_bxfx3, pf2_bxfx3), dim=2)

    xy_f0 = xy_bxpx2[:, faces_fx3[:, 0], :]
    xy_f1 = xy_bxpx2[:, faces_fx3[:, 1], :]
    xy_f2 = xy_bxpx2[:, faces_fx3[:, 2], :]
    points2d_bxfx6 = torch.cat((xy_f0, xy_f1, xy_f2), dim=2)

    v01_bxfx3 = pf1_bxfx3 - pf0_bxfx3
    v02_bxfx3 = pf2_bxfx3 - pf0_bxfx3

    normal_bxfx3 = torch.cross(v01_bxfx3, v02_bxfx3, dim=2)

    return points3d_bxfx9, points2d_bxfx6, normal_bxfx3

class Renderer(nn.Module):

    def __init__(self, height, width, filtering='bilinear', mode='texture'):
        super().__init__()

        assert mode in ['texture', 'vc', 'alpha']
        
        self.height = height
        self.width = width
        self.filtering = filtering
        self.mode = mode

    def forward(self, points, uv_bxpx2, texture_bx3xthxtw, ft_fx3=None, background_image=None, return_hardmask=False, delta=None):

        points_bxpx3, faces_fx3 = points
        
        if ft_fx3 is None:
            ft_fx3 = faces_fx3
            
        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = ortho_projection(points_bxpx3, faces_fx3)

        # Detect front/back faces
        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]

        if self.mode == 'texture':
            assert texture_bx3xthxtw is not None
            c0 = uv_bxpx2[:, ft_fx3[:, 0], :]
            c1 = uv_bxpx2[:, ft_fx3[:, 1], :]
            c2 = uv_bxpx2[:, ft_fx3[:, 2], :]
            mask = torch.ones_like(c0[:, :, :1])
            uv_bxfx9 = torch.cat((c0, mask, c1, mask, c2, mask), dim=2)
            rasterizer_input = uv_bxfx9
            
        elif self.mode == 'vc':
            assert texture_bx3xthxtw is not None
            colors_bxpx3 = texture_bx3xthxtw
            c0 = colors_bxpx3[:, faces_fx3[:, 0], :]
            c1 = colors_bxpx3[:, faces_fx3[:, 1], :]
            c2 = colors_bxpx3[:, faces_fx3[:, 2], :]
            mask = torch.ones_like(c0[:, :, :1])
            color_bxfx12 = torch.cat((c0, mask, c1, mask, c2, mask), dim=2)
            rasterizer_input = color_bxfx12
            
        else: # alpha
            #assert texture_bx3xthxtw is None
            rasterizer_input = torch.ones((points_bxpx3.shape[0], faces_fx3.shape[0], 3), device=points_bxpx3.device)
        
        if delta is None:
            delta = 7000
            
        if kaolin.__version__ == '0.1.0':
            imfeat, improb_bxhxwx1 = linear_rasterizer(
                self.height,
                self.width,
                points3d_bxfx9,
                points2d_bxfx6,
                normalz_bxfx1,
                rasterizer_input,
                0.02, # Expand
                30, # knum
                1000, # multiplier
                delta, # delta (default 7000)
            )
        else:
            imfeat, improb_bxhxwx1, _ = kaolin.render.mesh.dibr_rasterization(
                self.height,
                self.width,
                points3d_bxfx9.view(*points3d_bxfx9.shape[:2], 3, 3)[..., 2].contiguous(),
                points2d_bxfx6.view(*points2d_bxfx6.shape[:2], 3, 2),
                rasterizer_input.view(*rasterizer_input.shape[:2], 3, -1),
                normalz_bxfx1.squeeze(2),
                sigmainv=delta,
                boxlen=0.02,
                knum=30,
                multiplier=1000,
            )
            improb_bxhxwx1 = improb_bxhxwx1.unsqueeze(-1)
        
        if self.mode == 'vc':
            imrender = imfeat[:, :, :, :-1]
            hardmask = imfeat[:, :, :, -1:]
            
        elif self.mode == 'texture':
            imtexcoords = imfeat[:, :, :, :2]
            hardmask = imfeat[:, :, :, 2:3].detach()
            imrender = fragmentshader(imtexcoords, texture_bx3xthxtw, hardmask,
                                      filtering=self.filtering, background_image=background_image)
        else: # alpha
            hardmask = imfeat[:, :, :, :1]
            imrender = None
            

        if return_hardmask:
            improb_bxhxwx1 = hardmask
        return imrender, improb_bxhxwx1
