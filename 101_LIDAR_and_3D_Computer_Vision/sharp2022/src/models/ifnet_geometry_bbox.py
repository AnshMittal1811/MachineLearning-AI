import torch
import torch.nn as nn
import torch.nn.functional as F

class IFNetGeometryBbox(nn.Module):

    def __init__(self, hidden_dim=256):
        super(IFNetGeometryBbox, self).__init__()
        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = displacments 


        ### pose eatimation branch
        self.conv_in_pose = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')
        self.conv_0_pose = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')
        self.conv_0_1_pose = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')
        self.conv_1_pose = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')
        self.conv_1_1_pose = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')
        self.conv_2_pose = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')
        self.conv_2_1_pose = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3_pose = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3_1_pose = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')

        self.conv_in_bn_pose = nn.BatchNorm3d(16)
        self.conv0_1_bn_pose = nn.BatchNorm3d(32)
        self.conv1_1_bn_pose = nn.BatchNorm3d(64)
        self.conv2_1_bn_pose = nn.BatchNorm3d(128)
        self.conv3_1_bn_pose = nn.BatchNorm3d(128)
        self.maxpool_pose = nn.MaxPool3d(2)

        self.fc_0_pose = nn.Conv1d(128, 128, 1)
        self.fc_1_pose = nn.Conv1d(128, 6, 1)
        self.fc_2_pose = nn.Conv1d(6, 6, 1)



    def forward(self, x):
        b = x.size(0)
        x = x.unsqueeze(1)

        ### Pose estimation branch
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

        net_pose = torch.reshape(net_pose, (b, 128, -1 ))

        net_pose = self.actvn(self.fc_0_pose(net_pose))
        net_pose, _ = net_pose.max(dim=2)
        net_pose = self.actvn(self.fc_1_pose(net_pose.unsqueeze(2)))
        net_pose = self.fc_2_pose(net_pose)
        out_pose = net_pose.squeeze(2)
        

        return out_pose