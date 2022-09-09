import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.normalization import BatchNorm_StandingStats



def get_batchnorm_layer(opt):
    norm_G = opt.norm_G.split(":")[1]
    if norm_G == "batch":
        norm_layer = nn.BatchNorm2d
    elif norm_G == "spectral_instance":
        norm_layer = nn.InstanceNorm2d
    elif norm_G == "spectral_batch":
        norm_layer = nn.BatchNorm2d
    elif norm_G == "spectral_batchstanding":
        norm_layer = BatchNorm_StandingStats

    return norm_layer


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


class Unet_Cost_Volume(nn.Module):
    """
    A UNet model with a tiny Cost Volume in the bottom of the model.
    """

    def __init__(self, num_filters=32, channels_in=3, channels_out=3, use_tanh=False, opt=None, explicit_depth=False):
        super(Unet_Cost_Volume, self).__init__()
        self.opt = opt
        norm_layer = get_batchnorm_layer(opt)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, num_filters, 3, 2, 1),
            norm_layer(num_filters),
            nn.LeakyReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, 3, 2 ,1),
            norm_layer(num_filters * 2),
            nn.LeakyReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 1),
            norm_layer(num_filters * 4),
            nn.LeakyReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 1),
            norm_layer(num_filters * 8),
            nn.LeakyReLU(True),
        )

        self.feature = nn.Sequential(
            nn.Conv2d(num_filters * 8, num_filters * 2, 3, 1, 1),            
        )

        self.CostReg = nn.Sequential(
            nn.Conv3d(num_filters * 2, num_filters, 3, 1, 1),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(True),
            nn.Conv3d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(True),
            nn.Conv3d(num_filters,num_filters//2, 3, 1, 1)
        )
        self.CostProb = nn.Sequential(
            nn.BatchNorm3d(num_filters//2),
            nn.ReLU(True),
            nn.Conv3d(num_filters//2, 1, 3, 1, 1),
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        if explicit_depth:
            self.conv5 = nn.Sequential(
                nn.Conv2d(num_filters * 2 + 1, num_filters * 4, 3, 1, 1),
                norm_layer(num_filters * 4),
                nn.ReLU(True)
                )
        else:
            self.conv5 = nn.Sequential(
                nn.Conv2d(num_filters * 2 + num_filters//2 + 1, num_filters * 4, 3, 1, 1),
                norm_layer(num_filters * 4),
                nn.ReLU(True)
            )

        self.conv6 = nn.Sequential(
            nn.Conv2d(num_filters * 4 * 2, num_filters * 2, 3, 1, 1),
            norm_layer(num_filters * 2),
            nn.ReLU(True)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(num_filters * 4, num_filters, 3, 1, 1),
            norm_layer(num_filters),
            nn.ReLU(True)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1),
            norm_layer(num_filters),
            nn.ReLU(True),
            nn.Conv2d(num_filters, channels_out, 3, 1, 1)
        )
        self.explicit_depth = explicit_depth
        self.num_filters = num_filters

    def forward(self, input_img, proj_matrices, depth_values):
        """
        Assume input_img is a list, where  len(x) is the number of views and x[0].shape = [B, C, H, W]
        proj_matrices: a list of [Bs, nv, 4, 4]
        """
        num_depth = depth_values.shape[-1]
        num_filters = self.num_filters
        B, nv, C, H, W = input_img.shape
        input_img = input_img.view(-1, C, H, W) # shape Bxnv, C, H, W
        depths_values  = depth_values.unsqueeze(1).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)# bs x 1 x 1 x num_depths x 1 x 1
        e1 =  self.conv1(input_img)  # shape Bxnv, num_filters, H//2, W//2
        e2 = self.conv2(e1) # Bxnv, num_filters * 2, H//4, W//4
        e3 = self.conv3(e2) # Bxnv, num_filters * 4, H//8, W//8
        e4 = self.conv4(e3) # Bxnv, num_filters * 8, H//16, W//16

        feature = self.feature(e4) # Bxnv, num_filters * 2, H//16, W //16
        permuted_feature = feature.view(B, nv, num_filters*2, H//16, W//16)
        ref_volume = feature.view(B, nv, num_filters*2, H//16, W//16).unsqueeze(3).repeat(1, 1, 1, num_depth, 1, 1) # B, nv, num_filters *2, num_depth, H//16, W//16
        # volume_sum = ref_volume
        # volume_sq_sum =ref_volume ** 2 
        volume_sum = []
        volume_sq_sum = []
        for i in range(nv):
            volume_sum.append(ref_volume[:, i])
            volume_sq_sum.append(ref_volume[:, i]**2)
            
        for ref_index in range(nv):
            for  src_index in range(nv):

                if  ref_index == src_index:
                    continue
                else:
                    warped_volume = homo_warping(permuted_feature[:, src_index], proj_matrices[:, src_index], proj_matrices[:, ref_index], depth_values)
                    volume_sum[ref_index] = volume_sum[ref_index] + warped_volume
                    volume_sq_sum[ref_index] = volume_sq_sum[ref_index] +warped_volume ** 2 # bsx C x num_depths x H x W
        volume_sum = torch.stack(volume_sum, dim=1)  # bsx num_inputs x C x num_depths x H x W
        volume_sq_sum = torch.stack(volume_sq_sum, dim=1)
        volume_variance = volume_sq_sum.div_(nv).sub_(volume_sum.div_(nv).pow_(2))
        cost_reg = self.CostReg(volume_variance.view(-1, *volume_variance.shape[2:] )) # Bxnum_inputs, num_filters/2, num_depths,  H//16, W//16
        prob_volume = F.softmax(self.CostProb(cost_reg), dim=2)  # B*num_inputs, 1, num_depths,  H//16, W//16
        num_inputs = nv
        depths_values = depths_values.repeat(1, num_inputs,  1, 1, 1, 1)
        depths_values = depths_values.view(-1, *depths_values.shape[2:]) # bs*num_inputs, 1 , num_depths, 1, 1
        depth = torch.sum(prob_volume * depths_values, dim=2)
        depth = (depth - self.opt.min_z) / (self.opt.max_z - self.opt.min_z)
        com_feature =torch.sum(cost_reg * prob_volume, 2)
        combine_feature = torch.cat((com_feature, feature, depth), dim=1)
        # depths = depths.view(-1, *depths.shape[2:])
        # calculated_depth = torch.sum()
        d4_ = self.conv5(self.up(combine_feature)) # Bxnv, num_filters * 4, H//8, W//8
        d4 = torch.cat((d4_, e3), dim=1) # Bxnv, num_filters * 8, H//8, W//8
        d3_ = self.conv6(self.up(d4)) # Bxnv, num_filters * 2, H//4, W//4
        d3 = torch.cat((d3_, e2), dim=1) # Bxnv, num_filters * 4, H//4, W//4
        d2_ = self.conv7(self.up(d3)) # Bxnv, num_filters, H//2, W//2
        d2 = torch.cat((d2_, e1), dim=1) # Bxnv, num_filters * 2, H//2, W//2
        d1 =  self.conv8(self.up(d2)) # Bxnv, 2, H, W

        return d1

