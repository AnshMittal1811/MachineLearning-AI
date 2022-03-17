import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..DataTools.Loaders import _add_batch_one
from .modules import Attention


class _CoarseFlow(nn.Module):
    """
    Coarse Flow Network in MCT
    |----------------------|
    |    Input two frame   |
    |----------------------|
    | Conv k5-n24-s2, ReLu |
    |----------------------|
    | Conv k3-n24-s1, ReLu |
    |----------------------|
    | Conv k5-n24-s2, ReLu |
    |----------------------|
    | Conv k3-n24-s1, ReLu |
    |----------------------|
    | Conv k3-n32-s1, Tanh |
    |----------------------|
    |   Pixel Shuffle x4   |
    |----------------------|
    """
    def __init__(self, input_channel=1):
        super(_CoarseFlow, self).__init__()
        self.channel = input_channel
        self.conv1 = nn.Conv2d(input_channel * 2, 24, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(24, 24, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(24, 32, 3, stride=1, padding=1)
        self.pix = nn.PixelShuffle(4)

    def forward(self, frame_t, frame_tp1):
        input = torch.cat([frame_t, frame_tp1], dim=1)
        return self.pix(
            self.conv5(
                F.relu(self.conv4(
                    F.relu(self.conv3(
                        F.relu(self.conv2(
                            F.relu(self.conv1(input))
                        ))
                    ))
                ))
            )
        )


class _FineFlow(nn.Module):
    """
    Fine Flow Network in MCT
    |----------------------|
    |    Input two frame   |
    |----------------------|
    | Conv k5-n24-s2, ReLu |
    |----------------------|
    | Conv k3-n24-s1, ReLu |
    |----------------------|
    | Conv k3-n24-s1, ReLu |
    |----------------------|
    | Conv k3-n24-s1, ReLu |
    |----------------------|
    |  Conv k3-n8-s1, Tanh |
    |----------------------|
    |   Pixel Shuffle x2   |
    |----------------------|
    """
    def __init__(self, input_channel=1):
        super(_FineFlow, self).__init__()
        self.channel = input_channel
        self.conv1 = nn.Conv2d(input_channel * 3 + 2, 24, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(24, 8, 3, stride=1, padding=1)
        self.pix = nn.PixelShuffle(2)

    def forward(self, frame_t, frame_tp1, flow, coarse_frame_tp1):
        input = torch.cat([frame_t, frame_tp1, flow, coarse_frame_tp1], dim=1)
        return self.pix(
            self.conv5(
                F.relu(self.conv4(
                    F.relu(self.conv3(
                        F.relu(self.conv2(
                            F.relu(self.conv1(input))
                        ))
                    ))
                ))
            )
        )


class Advector(nn.Module):
    """
    According to the PyTorch documentation
    The X axis is positively toward left and the Y-axis is positively toward upward
    """
    def __init__(self):
        super(Advector, self).__init__()
        self.std_theta = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32)

    def forward(self, frame_t, vectors):
        N, C, H, W = frame_t.size()
        vectors[:, 0] = 2 * vectors[:, 0] / W
        vectors[:, 1] = 2 * vectors[:, 1] / H
        if isinstance(frame_t.data, torch.cuda.FloatTensor):
            std_theta = Variable(torch.from_numpy(np.repeat(self.std_theta, N, axis=0))).cuda()
        else:
            std_theta = Variable(torch.from_numpy(np.repeat(self.std_theta, N, axis=0)))
        for i in range(N):
            std_theta[i, :, 2] = vectors[i]
        if isinstance(frame_t.data, torch.cuda.FloatTensor):
            affine = F.affine_grid(std_theta, frame_t.size()).cuda()
        else:
            affine = F.affine_grid(std_theta, frame_t.size())
        return F.grid_sample(frame_t, affine)


class Warp(nn.Module):
    """
    Warp Using Optical Flow
    """
    def __init__(self):
        super(Warp, self).__init__()
        self.std_theta = np.eye(2, 3, dtype=np.float32).reshape((1, 2, 3))

    def forward(self, frame_t, flow_field):
        """
        :param frame_t: input batch of images (N x C x IH x IW)
        :param flow_field: flow_field with shape(N x 2 x OH x OW)
        :return: output Tensor
        """
        N, C, H, W = frame_t.size()
        std_theta = torch.from_numpy(np.repeat(self.std_theta, N, axis=0))
        if isinstance(frame_t.data, torch.cuda.FloatTensor):
            std = F.affine_grid(std_theta, frame_t.size()).cuda()
        else:
            std = F.affine_grid(std_theta, frame_t.size())
        flow_field[:, 0, :, :] = flow_field[:, 0, :, :] / W
        flow_field[:, 1, :, :] = flow_field[:, 1, :, :] / H
        return F.grid_sample(frame_t, std + flow_field.permute(0, 2, 3, 1))


class FlowField(nn.Module):
    """
    The final Fine Flow
    """
    def __init__(self):
        super(FlowField, self).__init__()
        self.coarse_net = _CoarseFlow()
        self.fine_net = _FineFlow()
        self.warp = Warp()

    def forward(self, frame_t, frame_tp1):
        coarse_flow = self.coarse_net(frame_t, frame_tp1)
        coarse_frame_tp1 = self.warp(frame_t, coarse_flow)
        return self.fine_net(frame_t, frame_tp1, coarse_flow, coarse_frame_tp1) + coarse_flow


class _CoarseFlowNoStride(nn.Module):
    """
    Coarse Flow Network in MCT without Stride
    |----------------------|
    |    Input two frame   |
    |----------------------|
    | Conv k5-n32-s1, ReLu |
    |----------------------|
    | Conv k3-n32-s1, ReLu |
    |----------------------|
    | Conv k5-n32-s1, ReLu |
    |----------------------|
    | Conv k3-n32-s1, ReLu |
    |----------------------|
    | Conv k3-n32-s1, Tanh |
    |----------------------|
    |   Pixel Shuffle x4   |
    |----------------------|
    """
    def __init__(self, input_channel=1):
        super(_CoarseFlowNoStride, self).__init__()
        self.channel = input_channel
        self.conv1 = nn.Conv2d(input_channel * 2, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(32, 2, 5, stride=1, padding=2)

    def forward(self, frame_t, frame_tp1):
        input = torch.cat([frame_t, frame_tp1], dim=1)
        return self.conv5(
                F.relu(self.conv4(
                    F.relu(self.conv3(
                        F.relu(self.conv2(
                            F.relu(self.conv1(input))
                        ))
                    ))
                ))
            )


class _FineFlowNoStride(nn.Module):
    """
    Fine Flow Network in MCT without Stride
    |----------------------|
    |    Input two frame   |
    |----------------------|
    | Conv k5-n24-s1, ReLu |
    |----------------------|
    | Conv k3-n24-s1, ReLu |
    |----------------------|
    | Conv k3-n24-s1, ReLu |
    |----------------------|
    | Conv k3-n24-s1, ReLu |
    |----------------------|
    |  Conv k3-n8-s1, Tanh |
    |----------------------|
    |   Pixel Shuffle x2   |
    |----------------------|
    """
    def __init__(self, input_channel=1):
        super(_FineFlowNoStride, self).__init__()
        self.channel = input_channel
        self.conv1 = nn.Conv2d(input_channel * 3 + 2, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 2, 3, stride=1, padding=1)

    def forward(self, frame_t, frame_tp1, flow, coarse_frame_tp1):
        input = torch.cat([frame_t, frame_tp1, flow, coarse_frame_tp1], dim=1)
        return self.conv5(
                F.relu(self.conv4(
                    F.relu(self.conv3(
                        F.relu(self.conv2(
                            F.relu(self.conv1(input))
                        ))
                    ))
                ))
            )


class FlowFielsNoStride(nn.Module):
    """
    The final Fine Flow
    """
    def __init__(self):
        super(FlowFielsNoStride, self).__init__()
        self.coarse_net = _CoarseFlowNoStride()
        self.fine_net = _FineFlowNoStride()
        self.warp = Warp()

    def forward(self, frame_t, frame_tp1):
        coarse_flow = self.coarse_net(frame_t, frame_tp1)
        coarse_frame_tp1 = self.warp(frame_t, coarse_flow)
        return self.fine_net(frame_t, frame_tp1, coarse_flow, coarse_frame_tp1) + coarse_flow


class Affine_Translation(nn.Module):
    def __init__(self, input_channel=1, freedom_degree=2):
        super(Affine_Translation, self).__init__()
        self.channel = input_channel
        self.conv1 = nn.Conv2d(input_channel * 2, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(16, 16, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32, freedom_degree)

    def forward(self, frame_t, frame_tp1):
        input = torch.cat([frame_t, frame_tp1], dim=1)
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = self.linear(x.view(x.size()[:2]))
        return x


class AdvectorFCN(nn.Module):
    """
        Discriminator with attention module
        """

    def __init__(self, input_nc=1, ndf=16, output_nc=2, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, feature_channels=16,
                 down_samples=2):
        super(AdvectorFCN, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        pixelS_net = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, output_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
        ]

        self.pixelD = nn.Sequential(*pixelS_net)
        self.attention = Attention(input_channel=input_nc, feature_channels=feature_channels, down_samples=down_samples)

    def forward(self, input, train=False, is_attention=False):
        pixel_level_pred = self.pixelD(input)
        if train:
            attention_map = self.attention(input)
        else:
            attention_map = self.attention(input).detach()
        weighted_pred = torch.mul(pixel_level_pred, attention_map)
        weighted_sum = torch.sum(weighted_pred.view(weighted_pred.size(0), weighted_pred.size(1), -1), dim=2)
        attention_sum = torch.sum(attention_map.view(attention_map.size(0), attention_map.size(1), -1), dim=2)
        final_pred = torch.div(weighted_sum, attention_sum)
        return_tuple = final_pred
        if train:
            return final_pred, pixel_level_pred
        elif is_attention:
            return final_pred, pixel_level_pred, attention_map
        else:
            return return_tuple


