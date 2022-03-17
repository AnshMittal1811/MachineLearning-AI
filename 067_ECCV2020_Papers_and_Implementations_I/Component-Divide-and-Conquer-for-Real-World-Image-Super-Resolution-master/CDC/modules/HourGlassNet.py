import math
import torch
import torch.nn as nn
from . import block as B


class ResidualBlock(nn.Module):
    """
    Residual BLock For SR without Norm Layer
    conv 1*1
    conv 3*3
    conv 1*1
    """
    def __init__(self, ch_in=64, ch_out=128, in_place=True):
        super(ResidualBlock, self).__init__()

        conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        conv2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        conv3 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        relu1 = nn.LeakyReLU(0.2, inplace=in_place)
        relu2 = nn.LeakyReLU(0.2, inplace=in_place)

        self.res = nn.Sequential(conv1, relu1, conv2, relu2, conv3)
        if ch_in != ch_out:
            self.identity = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        else:
            def identity(tensor):
                return tensor

            self.identity = identity

    def forward(self, x):
        res = self.res(x)
        x = self.identity(x)
        return torch.add(x, res)


class TopDownBlock(nn.Module):
    """
    Top to Down Block for HourGlass Block
    Consist of ConvNet Block and Pooling
    """
    def __init__(self, ch_in=64, ch_out=64, res_type='res'):
        super(TopDownBlock, self).__init__()
        if res_type == 'rrdb':
            self.res_block = B.RRDB(ch_in, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA')
        else:
            self.res_block = ResidualBlock(ch_in=ch_in, ch_out=ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.res_block(x)
        return self.pool(x), x


class BottomUpBlock(nn.Module):
    """
    Bottom Up Block for HourGlass Block
    Consist of ConvNet Block and Upsampling Block
    """
    def __init__(self, ch_in=64, ch_out=64, res_type='res'):
        super(BottomUpBlock, self).__init__()
        if res_type == 'rrdb':
            self.res_block = B.RRDB(ch_in, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA')
        else:
            self.res_block = ResidualBlock(ch_in=ch_in, ch_out=ch_out)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x, res):
        x = self.upsample(x)
        return self.res_block(x + res)


class HourGlassBlock(nn.Module):
    """
    Hour Glass Block for SR Model
    """
    def __init__(self, res_type='res', n_mid=2):
        super(HourGlassBlock, self).__init__()
        self.down1 = TopDownBlock(64, 128, res_type=res_type)
        self.down2 = TopDownBlock(128, 128, res_type=res_type)
        self.down3 = TopDownBlock(128, 256, res_type=res_type)
        self.down4 = TopDownBlock(256, 256, res_type=res_type)

        res_block = []
        for i in range(n_mid):
            res_block.append(ResidualBlock(256, 256))
        self.mid_res = nn.Sequential(*res_block)

        self.up1 = BottomUpBlock(256, 256, res_type=res_type)
        self.up2 = BottomUpBlock(256, 128, res_type=res_type)
        self.up3 = BottomUpBlock(128, 128, res_type=res_type)
        self.up4 = BottomUpBlock(128, 64, res_type=res_type)

    def forward(self, x):
        x, res1 = self.down1(x)
        x, res2 = self.down2(x)
        x, res3 = self.down3(x)
        x, res4 = self.down4(x)

        x = self.mid_res(x)

        x = self.up1(x, res4)
        x = self.up2(x, res3)
        x = self.up3(x, res2)
        x = self.up4(x, res1)
        return x

class HourGlassNet(nn.Module):
    """
    Hour Glass SR Model
    """
    def __init__(self, in_nc=3, out_nc=3, upscale=4, nf=64, res_type='res', n_mid=2, n_HG=3, act_type='leakyrelu'):
        super(HourGlassNet, self).__init__()

        self.n_HG = n_HG
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.conv_in = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='leakyrelu')
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        if upscale == 3:
            upsampler = B.pixelshuffle_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [B.pixelshuffle_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        upsample_block = nn.Sequential(LR_conv, *upsampler, HR_conv0, HR_conv1)

        for i in range(n_HG):
            HG_block = HourGlassBlock(res_type=res_type, n_mid=n_mid)
            setattr(self, 'HG_%d' % i, HG_block)
            setattr(self, 'upsample_%d' % i, upsample_block)

    def forward(self, x):
        x = self.conv_in(x)
        result = []
        for i in range(self.n_HG):
            x = getattr(self, 'HG_%d' % i)(x)
            out = getattr(self, 'upsample_%d' % i)(x)
            result.append(out)

        return result


# if __name__ == '__main__':
#     model = HourGlassNet(upscale=2, n_HG=2)
#     from TorchNet.tools import calculate_parameters
#     p = calculate_parameters(model)
#     print(p)
# #     input = torch.FloatTensor(1, 3, 16, 16).cuda()
# #     result = model(input)
# #     for tensor in result:
# #         print(tensor.shape)























