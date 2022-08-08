import torch
import torch.nn as nn


class WarpTransform(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.warp_cropped_car = opt.warp_cropped_car

    def forward(self, x, data):
        # crop out black regions for StyleGAN cars
        if self.warp_cropped_car:
            B, C, H, W = x.shape
            black = -torch.ones_like(x)
            x = x[:, :, H // 8:7 * H // 8, :]

        deform = data['warp_grid'].to(x.device).type(x.type())
        warped = nn.functional.grid_sample(x, deform, align_corners=False, padding_mode="reflection")

        # paste the warped car back into the black canvas
        if self.warp_cropped_car:
            black[:, :, H // 8:7 * H // 8, :] = warped
            warped = black

        return warped
