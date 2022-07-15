"""
A stereo matching network (GCNet) from "End-to-End Learning of Geometry and Context for Deep Stereo Regression".
Ref: https://github.com/wyf2017/DSMnet/blob/master/models/gcnet.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from .gcnet_conv import net_init, conv2d_bn, conv_res, conv3d_bn, deconv3d_bn
from .gcnet_fun import myAdd3d


flag_bias_t = True
flag_bn = True
activefun_t = nn.ReLU(inplace=True)


class GCNet(nn.Module):
    def __init__(self, maxdisparity=192):
        super(GCNet, self).__init__()
        self.D = maxdisparity // 2
        self.count_levels = 1
        self.layer2d = feature2d(32)
        self.layer3d = feature3d(32)

        net_init(self)

    def forward(self, inputs, mode='train'):
        imL, imR = inputs['left_rgb'], inputs['right_rgb']
        assert imL.shape == imR.shape
        # Extract 2D features for left and right images
        fL = self.layer2d(imL)
        fR = self.layer2d(imR)
        # Construct cost volume
        n, F, h, w = fL.shape # bsize, feat_dim, h//2, w//2
        xL = Variable(torch.zeros(n, F*2, self.D, h, w).type_as(fL.data))
        xL[:, :, 0] = torch.cat([fL, fR], 1)
        for i in range(1, self.D):
            xL[:, :F, i] = fL
            xL[:, F:, i, :, i:] = fR[:, :, :, :-i]
        # Perform 3DCNN
        oL = self.layer3d(xL, mode)[:, :, :imL.shape[-2], :imL.shape[-1]]
        return oL


class feature2d(nn.Module):
    def __init__(self, num_F=32):
        super(feature2d, self).__init__()
        self.inplanes = 32
        self.F = num_F

        self.conv1 = conv2d_bn(3, 32, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.block1 = conv_res(32, 32, blocks=8, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.conv2(x)
        return x


class feature3d(nn.Module):
    def __init__(self, num_F=32):
        super(feature3d, self).__init__()
        self.F = num_F

        self.l19 = conv3d_bn(self.F*2, self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l20 = conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.l21 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l22 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l23 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.l24 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l25 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l26 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.l27 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l28 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l29 = conv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.l30 = conv3d_bn(self.F*2, self.F*4, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l31 = conv3d_bn(self.F*4, self.F*4, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l32 = conv3d_bn(self.F*4, self.F*4, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.l33 = deconv3d_bn(self.F*4, self.F*2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l34 = deconv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l35 = deconv3d_bn(self.F*2, self.F*2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l36 = deconv3d_bn(self.F*2, self.F, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.l37 = deconv3d_bn(self.F, 1, kernel_size=3, stride=2, bn=False, activefun=None)
        self.softmax = nn.Softmax2d()

    def forward(self, x, mode="train"):
        x18 = x
        x21 = self.l21(x18)
        x24 = self.l24(x21)
        x27 = self.l27(x24)
        x30 = self.l30(x27)
        x31 = self.l31(x30)
        x32 = self.l32(x31)
        if(mode=="test"): del x30, x31

        # x32 x29
        x29 = self.l29(self.l28(x27))
        if(mode=="test"): del x27
        x33 = myAdd3d(self.l33(x32), x29)
        if(mode=="test"): del x32, x29

        # x33 x26
        x26 = self.l26(self.l25(x24))
        if(mode=="test"): del x24
        x34 = myAdd3d(self.l34(x33), x26)
        if(mode=="test"): del x33, x26

        # x34 x23
        x23 = self.l23(self.l22(x21))
        if(mode=="test"): del x21
        x35 = myAdd3d(self.l35(x34), x23)
        if(mode=="test"): del x34, x23

        # x35 x20
        x20 = self.l20(self.l19(x18))
        if(mode=="test"): del x, x18
        x36 = myAdd3d(self.l36(x35), x20)
        if(mode=="test"): del x35, x20

        # x36
        x37 = self.l37(x36)
        if(mode=="test"): del x36

        # x37
        out = self.softmax(-x37.squeeze(1))
        if(mode=="test"): del x37

        # out
        tmp = Variable(torch.arange(0, out.shape[1]).type_as(out.data))
        out = out.permute(0,2,3,1).matmul(tmp)

        return out.unsqueeze(1)
