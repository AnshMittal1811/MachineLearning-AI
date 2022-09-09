import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class blending_network(nn.Module):

    def __init__(self, num_imgs=3):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * num_imgs, 32, 3, 1, 1),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(48, 48, 3, 2, 1),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(96, 96, 3, 2, 1),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(96, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(True)
        )

        self.conv6 = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(128, 96, 3, 1, 1),
            nn.ReLU(True)
        )

        self.conv7 = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.ReLU(True)
        )

        self.conv8 = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(128, 48, 3, 1, 1),
            nn.ReLU(True)
        )

        self.conv9 = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.ReLU(True)
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(64, num_imgs, 3, 1, 1)
        )
        
    def forward(self, x):
        x = torch.stack(x, dim=1)
        bs, N, C, H, W = x.shape
        x = x.view(bs, -1, H, W)
        factor = 16
        padding_H = math.ceil( H / factor) * factor
        padding_W = math.ceil( W / factor) * factor
        x = F.pad(x, (0, padding_W - W, 0, padding_H - H), mode='constant', value=0)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        concat1 = torch.cat((conv6, conv4), dim=1)
        conv7 = self.conv7(concat1)
        concat2 = torch.cat((conv7,conv3), dim=1)
        conv8  = self.conv8(concat2)
        concat3 = torch.cat((conv8, conv2), dim=1)
        conv9 = self.conv9(concat3)
        feature = torch.cat((conv9, conv1), dim=1)

        out = self.conv10(feature)
        softmax = torch.nn.Softmax(dim=1)(out)

        out = x.contiguous().view(bs, N, C, padding_H, padding_W)
        out = out * softmax.unsqueeze(2)
        out_img = torch.sum(out, dim=1)
        return out_img[:, :, 0:H, 0:W]




















































