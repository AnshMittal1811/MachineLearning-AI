import torch
import torch.nn as nn
import torch.nn.functional as F


class IFNetGeometryPose(nn.Module):

    def __init__(self, hidden_dim=256):
        super(IFNetGeometryPose, self).__init__()
        # 128**3 res input
        self.actvn = nn.ReLU()

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = displacments

        # pose eatimation branch
        self.conv_in_pose = nn.Conv3d(
            1, 16, 3, padding=1, padding_mode='replicate')
        self.conv_0_pose = nn.Conv3d(
            16, 32, 3, padding=1, padding_mode='replicate')
        self.conv_0_1_pose = nn.Conv3d(
            32, 32, 3, padding=1, padding_mode='replicate')
        self.conv_1_pose = nn.Conv3d(
            32, 64, 3, padding=1, padding_mode='replicate')
        self.conv_1_1_pose = nn.Conv3d(
            64, 64, 3, padding=1, padding_mode='replicate')
        self.conv_2_pose = nn.Conv3d(
            64, 128, 3, padding=1, padding_mode='replicate')
        self.conv_2_1_pose = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3_pose = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3_1_pose = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')

        self.conv_in_bn_pose = nn.InstanceNorm3d(16)
        self.conv0_1_bn_pose = nn.InstanceNorm3d(32)
        self.conv1_1_bn_pose = nn.InstanceNorm3d(64)
        self.conv2_1_bn_pose = nn.InstanceNorm3d(128)
        self.conv3_1_bn_pose = nn.InstanceNorm3d(128)
        self.maxpool_pose = nn.MaxPool3d(2)

        self.fc_0_pose = nn.Conv1d(128, 128, 1)
        self.fc_1_pose = nn.Conv1d(128, 64, 1)
        self.fc_2_pose = nn.Conv1d(64, 17*3, 1)

    def forward(self, p, x):
        b = x.size(0)
        x = x.unsqueeze(1)

        # Pose estimation branch
        net_pose = self.actvn(self.conv_in_pose(x))
        net_pose = self.conv_in_bn_pose(net_pose)
        net_pose = self.maxpool_pose(net_pose)

        net_pose = self.actvn(self.conv_0_pose(net_pose))
        net_pose = self.actvn(self.conv_0_1_pose(net_pose))
        net_pose = self.conv0_1_bn_pose(net_pose)
        net_pose = self.maxpool_pose(net_pose)

        net_pose = self.actvn(self.conv_1_pose(net_pose))
        net_pose = self.actvn(self.conv_1_1_pose(net_pose))
        net_pose = self.conv1_1_bn_pose(net_pose)
        net_pose = self.maxpool_pose(net_pose)

        net_pose = self.actvn(self.conv_2_pose(net_pose))
        net_pose = self.actvn(self.conv_2_1_pose(net_pose))
        net_pose = self.conv2_1_bn_pose(net_pose)
        net_pose = self.maxpool_pose(net_pose)

        net_pose = self.actvn(self.conv_3_pose(net_pose))
        net_pose = self.actvn(self.conv_3_1_pose(net_pose))
        net_pose = self.conv3_1_bn_pose(net_pose)
        net_pose = self.maxpool_pose(net_pose)

        net_pose = torch.reshape(net_pose, (b, 128, -1))

        net_pose = self.actvn(self.fc_0_pose(net_pose))
        net_pose, _ = net_pose.max(dim=2)
        net_pose = self.actvn(self.fc_1_pose(net_pose.unsqueeze(2)))
        net_pose = self.fc_2_pose(net_pose)
        out_pose = net_pose.squeeze(2)

        return out_pose.reshape(b, -1, 3)
