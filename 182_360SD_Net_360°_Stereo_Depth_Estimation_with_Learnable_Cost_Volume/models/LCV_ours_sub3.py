from __future__ import print_function
import math

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from sub_ASPP import convbn, convbn_3d, feature_extraction


class forfilter(nn.Module):
    def __init__(self, inplanes):
        super(forfilter, self).__init__()

        self.forfilter1 = nn.Conv2d(1, 1, (7, 1), 1, (0, 0), bias=False)
        self.inplanes = inplanes

    def forward(self, x):

        out = self.forfilter1(
            F.pad(torch.unsqueeze(x[:, 0, :, :], 1),
                  pad=(0, 0, 3, 3),
                  mode='replicate'))
        for i in range(1, self.inplanes):
            out = torch.cat((out,
                             self.forfilter1(
                                 F.pad(torch.unsqueeze(x[:, i, :, :], 1),
                                       pad=(0, 0, 3, 3),
                                       mode='replicate'))), 1)

        return out


class disparityregression_sub3(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression_sub3, self).__init__()
        self.disp = Variable(torch.Tensor(
            np.reshape(np.array(range(maxdisp * 3)), [1, maxdisp * 3, 1, 1]) /
            3).cuda(),
                             requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               stride=1,
                               pad=1)

        self.conv3 = nn.Sequential(
            convbn_3d(inplanes * 2,
                      inplanes * 2,
                      kernel_size=3,
                      stride=2,
                      pad=1), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn_3d(inplanes * 2,
                      inplanes * 2,
                      kernel_size=3,
                      stride=1,
                      pad=1), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               padding=1,
                               output_padding=(1, 1, 1),
                               stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes *
                           2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=(0, 1, 1),
                               stride=2,
                               bias=False), nn.BatchNorm3d(inplanes))

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu,
                          inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)

        return out, pre, post


class LCV(nn.Module):
    def __init__(self, maxdisp):
        super(LCV, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # for loop of filter kernel
        self.forF = forfilter(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, up, down):

        refimg_fea = self.feature_extraction(up)  # reference image feature
        targetimg_fea = self.feature_extraction(down)  # target image feature

        # matching
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0],
                              refimg_fea.size()[1] * 2, self.maxdisp / 4 * 3,
                              refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp / 4 * 3):
            if i > 0:
                cost[:, :refimg_fea.size()[1],
                     i, :, :] = refimg_fea[:, :, :, :]
                cost[:, refimg_fea.size()[1]:,
                     i, :, :] = shift_down[:, :, :, :]
                shift_down = self.forF(shift_down)
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
                shift_down = self.forF(targetimg_fea)

        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(
                cost1,
                [self.maxdisp * 3,
                 up.size()[2], up.size()[3]],
                mode='trilinear'
            )
            cost2 = F.upsample(
                cost2,
                [self.maxdisp * 3,
                 up.size()[2], up.size()[3]],
                mode='trilinear')

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression_sub3(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression_sub3(self.maxdisp)(pred2)

        cost3 = F.upsample(
            cost3, [self.maxdisp * 3,
                    up.size()[2], up.size()[3]],
            mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparityregression_sub3(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3
