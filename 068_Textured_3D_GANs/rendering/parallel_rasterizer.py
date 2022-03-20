import torch
import torch.nn as nn

from .renderer import Renderer

class ParallelKaolinRasterizer(nn.Module):
    
    def __init__(self, image_size, mode='vc'):
        super().__init__()
        self.renderer = Renderer(image_size, image_size, mode=mode)
        self.image_size = image_size
        self.mode = mode
        self.sigma_factor = 1
        
    def set_mode(self, mode):
        self.renderer.mode = mode
        self.mode = mode

    def forward(self, points, uv_bxpx2, texture_bx3xthxtw, ft_fx3=None, background_image=None,
                return_hardmask=False, closure=None, image_size=None, **kwargs):
        
        if image_size is None:
            image_size = self.image_size
        self.renderer.width = image_size
        self.renderer.height = image_size
        
        delta = int(7000/self.sigma_factor)
        rgb, alpha = self.renderer(points, uv_bxpx2, texture_bx3xthxtw, ft_fx3, background_image, return_hardmask, delta=delta)
        rgb = rgb.permute(0, 3, 1, 2) if rgb is not None else None
        alpha = alpha.permute(0, 3, 1, 2)
        
        if closure is not None:
            # For parallel loss computation:
            # the closure is called inside the nn.DataParallel routine and is thus parallelized
            return closure(rgb, alpha, **kwargs)
        else:
            return rgb, alpha
        
    def set_sigma_mul(self, factor):
        self.sigma_factor = factor