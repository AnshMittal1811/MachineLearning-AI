# credit: https://github.com/milesial/Pytorch-UNet
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm='none'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm=='instance':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        elif norm=='batch':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        elif norm=='none':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm='none'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm=='instance':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif norm=='batch':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif norm=='none':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, single=False, norm='none'):
        super().__init__()
        print('down norm:', norm)

        if single:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                SingleConv(in_channels, out_channels, norm=norm)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, norm=norm)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, single=False, norm='none'):
        super().__init__()
        print('up norm:', norm)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if single:
                self.conv = SingleConv(in_channels, out_channels, in_channels // 2, norm=norm)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if single:
                self.conv = SingleConv(in_channels, out_channels, norm=norm)
            else:
                self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpSample(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, single=False, norm='none'):
        super().__init__()
        print('up norm:', norm)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if single:
                self.conv = SingleConv(in_channels, out_channels, in_channels // 2, norm=norm)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            if single:
                self.conv = SingleConv(in_channels, out_channels, norm=norm)
            else:
                self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

""" Full assembly of the parts to form the complete network """
class SmallUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, single=True, norm='none', render_scale=1):
        super(SmallUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        assert (render_scale==1 or render_scale==2)
        self.render_scale = render_scale

        self.inc = SingleConv(n_channels, 128, norm=norm)
        self.down1 = Down(128, 256, single=single, norm=norm)
        self.down2 = Down(256, 512, single=single, norm=norm)
        self.up1 = Up(512, 256, bilinear, single=single, norm=norm)
        self.up2 = Up(256, 128, bilinear, single=single, norm=norm)

        if render_scale==2:
            self.up3 = UpSample(128, 128, bilinear, single=False, norm=norm)

        self.outc = OutConv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        if self.render_scale==2:
            x = self.up3(x)
        logits = self.outc(x)
        
        return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, single=False, norm='none', render_scale=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        assert (render_scale == 1 or render_scale == 2)
        self.render_scale = render_scale

        self.inc = DoubleConv(n_channels, 64, norm=norm)
        self.down1 = Down(64, 128, single=single, norm=norm)
        self.down2 = Down(128, 256, single=single, norm=norm)
        self.down3 = Down(256, 512, single=single, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, single=single, norm=norm)
        self.up1 = Up(1024, 512 // factor, bilinear, single=single, norm=norm)
        self.up2 = Up(512, 256 // factor, bilinear, single=single, norm=norm)
        self.up3 = Up(256, 128 // factor, bilinear, single=single, norm=norm)
        self.up4 = Up(128, 64, bilinear, single=single, norm=norm)

        if render_scale==2:
            self.up5 = UpSample(64, 64, bilinear, single=False, norm=norm)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.render_scale == 2:
            x = self.up5(x)
        logits = self.outc(x)
        return logits

class NPBGUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, single=False, norm='none', render_scale=1):
        super(NPBGUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        assert (render_scale == 1 or render_scale == 2)
        self.render_scale = render_scale

        self.inc = DoubleConv(n_channels, 64, norm=norm)
        self.down1 = Down(64, 128-n_channels, single=single, norm=norm) # need to concat with additional input, so reduce feat dim here
        self.down2 = Down(128, 256-n_channels, single=single, norm=norm)

        self.down3 = Down(256, 512, single=single, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, single=single, norm=norm)
        self.up1 = Up(1024, 512 // factor, bilinear, single=single, norm=norm)
        self.up2 = Up(512, 256 // factor, bilinear, single=single, norm=norm)
        self.up3 = Up(256, 128 // factor, bilinear, single=single, norm=norm)
        self.up4 = Up(128, 64, bilinear, single=single, norm=norm)

        if render_scale==2:
            self.up5 = UpSample(64, 64, bilinear, single=False, norm=norm)

        self.outc = OutConv(64, n_classes)

    def forward(self, x_in_s):
        x_in_0, x_in_1, x_in_2 = x_in_s
        x1 = self.inc(x_in_0)
        x2 = self.down1(x1)
        x2 = torch.cat([x2, x_in_1], dim=1)
        x3 = self.down2(x2)
        x3 = torch.cat([x3, x_in_2], dim=1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.render_scale == 2:
            x = self.up5(x)
        logits = self.outc(x)
        return logits

class TwoLayersCNN(nn.Module):
    def __init__(self, n_channels, n_classes, norm='none'):
        if norm == 'instance':
            self.shader_2d = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(args.shader_output_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_classes, kernel_size=3, padding=1),
            )
        elif norm == 'batch':
            self.shader_2d = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(args.shader_output_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_classes, kernel_size=3, padding=1),
            )
        elif norm == 'none':
            self.shader_2d = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_classes, kernel_size=3, padding=1),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.shader_2d(x)