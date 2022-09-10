# Hu et al. Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries
# Original code from: https://github.com/JunjH/Revisiting_Single_Depth_Estimation

import torch
from torch import nn
from torch.nn import functional as F

from lib.config import config
from lib.layers import FrozenBatchNorm2d
from lib.structures import DepthMap
from .sobel import Sobel
from ..utils import ModuleResult


class DepthPrediction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        block_channel = self.get_block_channel_list()

        self.model = DepthPredictionBackbone(num_features=block_channel[-1], block_channel=block_channel)

        self.cos_loss = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel().to(config.MODEL.DEVICE)
        self.criterionL1 = F.l1_loss

    @staticmethod
    def get_block_channel_list():
        block_channel_map = {
            # resnet
            "50": [256, 512, 1024, 2048],
            "34": [64, 128, 256, 512],
            "18": [64, 128, 256, 512],
        }
        identifier = config.MODEL.BACKBONE.CONV_BODY.split('-')[1]
        block_channels = block_channel_map[identifier]

        return block_channels

    def forward(self, features, depth_target) -> ModuleResult:
        depth_pred, depth_feature = self.model(features)
        depth_return = [DepthMap(p_[0].cpu(), t_.get_intrinsic()) for p_, t_ in zip(depth_pred, depth_target)]
        depth_target = torch.stack([target.get_tensor() for target in depth_target]).float().to(config.MODEL.DEVICE).unsqueeze(1)

        results = {
            "prediction": depth_pred,
            "return": depth_return,
            "features": depth_feature
        }

        losses = {}

        if self.training:

            # Mask invalid depth pixels
            valid_masks = torch.stack([(depth.depth_map != 0.0).bool() for depth in depth_target], dim=0)
            valid_masks.unsqueeze_(1)

            grad_target = self.get_gradient(depth_target)
            grad_pred = self.get_gradient(depth_pred)

            grad_target_dx = grad_target[:, 0, :, :].contiguous().view_as(depth_target)
            grad_target_dy = grad_target[:, 1, :, :].contiguous().view_as(depth_target)
            grad_pred_dx = grad_pred[:, 0, :, :].contiguous().view_as(depth_target)
            grad_pred_dy = grad_pred[:, 1, :, :].contiguous().view_as(depth_target)

            ones = torch.ones(depth_target.size(0), 1, depth_target.size(2), depth_target.size(3)).float().to(
                config.MODEL.DEVICE)
            normal_target = torch.cat((-grad_target_dx, -grad_target_dy, ones), 1)
            normal_pred = torch.cat((-grad_pred_dx, -grad_pred_dy, ones), 1)

            loss_depth = torch.log(torch.abs(depth_target - depth_pred) + 0.5)[valid_masks].mean()
            loss_dx = torch.log(torch.abs(grad_target_dx - grad_pred_dx) + 0.5)[valid_masks].mean()
            loss_dy = torch.log(torch.abs(grad_target_dy - grad_pred_dy) + 0.5)[valid_masks].mean()
            loss_gradient = loss_dx + loss_dy
            loss_normal = torch.abs(1 - self.cos_loss(normal_pred, normal_target))[valid_masks.squeeze(1)].mean()

            loss_weight = config.MODEL.DEPTH2D.LOSS_WEIGHT

            losses = {
                "depth": loss_weight * loss_depth,
                "normal": loss_weight * loss_normal,
                "gradient": loss_weight * loss_gradient
            }

        return losses, results

    def inference(self, features):
        depth_pred, depth_feature = self.model(features)

        return depth_pred, depth_feature


class _UpProjection(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        norm_func = FrozenBatchNorm2d if config.MODEL.FIXNORM else nn.BatchNorm2d

        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm_func(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.bn1_2 = norm_func(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = norm_func(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)

        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class DepthPredictionBackbone(nn.Module):
    def __init__(self, num_features, block_channel):
        super().__init__()

        self.D = D(num_features)
        self.MFF = MFF(block_channel)
        self.R = R(block_channel)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = x
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))
        return out, torch.cat((x_decoder, x_mff), 1)


class D(nn.Module):
    def __init__(self, num_features=2048):
        super().__init__()
        norm_func = FrozenBatchNorm2d if config.MODEL.FIXNORM else nn.BatchNorm2d

        self.conv = nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = norm_func(num_features)

        self.up1 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up2 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up3 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up4 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])

        return x_d4


class MFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super().__init__()
        norm_func = FrozenBatchNorm2d if config.MODEL.FIXNORM else nn.BatchNorm2d

        self.up1 = _UpProjection(num_input_features=block_channel[0], num_output_features=16)
        self.up2 = _UpProjection(num_input_features=block_channel[1], num_output_features=16)
        self.up3 = _UpProjection(num_input_features=block_channel[2], num_output_features=16)
        self.up4 = _UpProjection(num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = norm_func(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class R(nn.Module):
    def __init__(self, block_channel):
        super().__init__()
        norm_func = FrozenBatchNorm2d if config.MODEL.FIXNORM else nn.BatchNorm2d

        num_features = 64 + block_channel[3] // 32
        self.conv0 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = norm_func(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm_func(num_features)

        self.conv2 = nn.Conv2d(num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)

        return x2
