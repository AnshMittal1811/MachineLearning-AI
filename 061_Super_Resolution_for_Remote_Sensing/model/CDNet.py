import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import build_backbone
from .decoder import build_decoder
from data_utils import get_transform

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class CDNet(nn.Module):
    def __init__(self, args, backbone='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3):
        super(CDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.transform = get_transform(convert=True, normalize=True)

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)
        self.decoder = build_decoder(f_c, BatchNorm)

        self.cbam0 = CBAM(64)
        self.cbam1 = CBAM(64)

        self.cbam2 = CBAM(64)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(256)
        self.cbam5 = CBAM(512)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, hr_img1, hr_img2):

        x_1, f2_1, f3_1, f4_1 = self.backbone(hr_img1)
        x_2, f2_2, f3_2, f4_2 = self.backbone(hr_img2)

        x1 = self.decoder(self.cbam5(x_1), self.cbam2(f2_1), self.cbam3(f3_1), self.cbam4(f4_1))
        x2 = self.decoder(self.cbam5(x_2), self.cbam2(f2_2), self.cbam3(f3_2), self.cbam4(f4_2))

        x1 = self.cbam0(x1)
        x2 = self.cbam0(x2)

        dist = F.pairwise_distance(x1, x2, keepdim=True)
        dist = F.interpolate(dist, size=hr_img1.shape[2:], mode='bicubic', align_corners=True)

        return dist

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()