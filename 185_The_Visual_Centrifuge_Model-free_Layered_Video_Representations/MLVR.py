import torch
import torch.nn as nn


class conv3d_bn(nn.Module):
    def __init__(self, in_ch, out_ch, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0)):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.s = s
        self.p = p
        self.conv3d = nn.Sequential(
            nn.Conv3d(self.in_ch,
                      self.out_ch,
                      kernel_size=self.k,
                      stride=self.s,
                      padding=self.p), nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv3d(x)


class trans3d_bn(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch=(64, 64),
                 k=(1, 1, 1),
                 s=(1, 1, 1),
                 p=(0, 0, 0)):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.s = s
        self.p = p
        self.trans3d = nn.Sequential(
            nn.ConvTranspose3d(self.in_ch,
                               self.out_ch[0],
                               kernel_size=self.k,
                               stride=self.s,
                               padding=self.p),
            nn.BatchNorm3d(self.out_ch[0]),
            nn.ReLU(inplace=True),
            conv3d_bn(self.out_ch[0],
                      self.out_ch[1],
                      k=(3, 5, 5),
                      s=(1, 1, 1),
                      p=(1, 2, 2))  # default kernel_size = 2,4,4 or 4,4,4
        )

    def forward(self, x):
        return self.trans3d(x)


class Mixed(nn.Module):
    def __init__(self, in_ch=192, out_ch=(64, 96, 128, 16, 32, 32)):
        super().__init__()
        self.branch0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[0])

        self.branch1_0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[1])
        self.branch1_1 = conv3d_bn(in_ch=out_ch[1],
                                   out_ch=out_ch[2],
                                   k=(3, 3, 3),
                                   p=(1, 1, 1))

        self.branch2_0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[3])
        self.branch2_1 = conv3d_bn(in_ch=out_ch[3],
                                   out_ch=out_ch[4],
                                   k=(3, 3, 3),
                                   p=(1, 1, 1))

        self.branch3_0 = nn.MaxPool3d(kernel_size=(3, 3, 3),
                                      stride=(1, 1, 1),
                                      padding=(1, 1, 1))
        self.branch3_1 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[5])

        self.output_channels = out_ch[0] + out_ch[2] + out_ch[
            4] + in_ch  # conv1, conv2, conv3, max1

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1_1(self.branch1_0(x))
        b2 = self.branch2_1(self.branch2_0(x))
        b3 = self.branch3_1(self.branch3_0(x))

        return torch.cat([b0, b1, b2, b3], 1)


# Model-Free Layered Video Representation
class MLVR(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.n_channels = n_channels
        self.n_channels = n_classes
        self.Pred_I3D = InceptionI3D(n_channels=n_channels)
        self.Pred_Decoder = Decoder(n_classes=n_classes)
        self.Corr_I3D = InceptionI3D(n_channels=n_classes * 3)
        self.Corr_Decorder = Decoder(n_classes=n_classes)

    def forward(self, x):
        outputs = dict()
        pred_temp = self.Pred_I3D(x)
        pred_inter = self.Pred_Decoder(pred_temp)
        corr_temp = self.Corr_I3D(pred_inter)
        corr = self.Corr_Decorder(corr_temp)

        output_layers = pred_inter + corr

        outputs['pred_temp'] = pred_temp
        outputs['pred_inter'] = pred_inter
        outputs['corr_temp'] = corr_temp
        outputs['corr'] = corr
        outputs['output_layers'] = output_layers

        return outputs


class Decoder(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        """
        Upconv1
        """
        self.up_5c = trans3d_bn(in_ch=1024,
                                out_ch=(64, 64),
                                k=(2, 4, 4),
                                s=(2, 2, 2),
                                p=(0, 1, 1))  # 64, 16, 14, 14
        self.up_4f = conv3d_bn(in_ch=832, out_ch=64)
        """
        Upconv2
        """
        self.up_5c4f = trans3d_bn(in_ch=128,
                                  out_ch=(64, 64),
                                  k=(4, 4, 4),
                                  s=(2, 2, 2),
                                  p=(1, 1, 1))  # 64, 32, 28, 28
        self.up_3c = conv3d_bn(in_ch=480, out_ch=64)
        """
        Upconv3
        """
        self.up_5c4f3c = trans3d_bn(
            in_ch=128, out_ch=(32, 32), k=(5, 4, 4), s=(1, 2, 2),
            p=(2, 1, 1))  # 32, 32, 56, 56, default kernel_size = 4
        self.up_2c = conv3d_bn(in_ch=192, out_ch=32)
        """
        Upconv4
        """
        self.up_5c4f3c2c = trans3d_bn(
            in_ch=64, out_ch=(32, 16), k=(5, 4, 4), s=(1, 2, 2),
            p=(2, 1, 1))  # 32, 32, 112, 112, default kernel_size = 4
        self.up_1a = conv3d_bn(in_ch=64, out_ch=16)
        """
        Upconv5
        """
        self.final_up = nn.ConvTranspose3d(32,
                                           32,
                                           kernel_size=(4, 4, 4),
                                           stride=(2, 2, 2),
                                           padding=(1, 1,
                                                    1))  # 32, 64, 224, 224
        self.out = nn.Conv3d(
            32,
            self.n_classes * 3,
            kernel_size=(5, 5, 5),
            stride=(1, 1, 1),
            padding=(2, 2,
                     2))  # n*3, 64, 224, 224, default kernel_size = 4,4,4

    def forward(self, inputs):
        """
        init
        """
        Mix5c = inputs['Mix5c']
        Mix4f = inputs['Mix4f']
        Mix3c = inputs['Mix3c']
        conv2c = inputs['Conv2c']
        conv1a = inputs['Conv1a']
        """
        Upconv1
        """
        Up_5c = self.up_5c(Mix5c)
        Up_4f = self.up_4f(Mix4f)
        Cat_5c4f = torch.cat([Up_5c, Up_4f], 1)
        """
        Upconv2
        """
        Up_5c4f = self.up_5c4f(Cat_5c4f)
        Up_3c = self.up_3c(Mix3c)
        Cat_5c4f3c = torch.cat([Up_5c4f, Up_3c], 1)
        """
        Upconv3
        """
        Up_5c4f3c = self.up_5c4f3c(Cat_5c4f3c)
        Up_2c = self.up_2c(conv2c)
        Cat_5c4f3c2c = torch.cat([Up_5c4f3c, Up_2c], 1)
        """
        Upconv4
        """
        Up_5c4f3c2c = self.up_5c4f3c2c(Cat_5c4f3c2c)
        Up_1a = self.up_1a(conv1a)
        Cat_final = torch.cat([Up_5c4f3c2c, Up_1a], 1)
        """
        Upconv5
        """
        Up_final = self.final_up(Cat_final)
        out = self.out(Up_final)
        """
        result
        """
        B, C, S, H, W = out.shape

        return out


class InceptionI3D(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        self.n_channels = n_channels
        """
        Conv3d_1a_7x7
        """
        self.conv3d_1a = conv3d_bn(n_channels,
                                   out_ch=64,
                                   k=(7, 7, 7),
                                   s=(2, 2, 2),
                                   p=(3, 3, 3))
        """
        MaxPool3d_2a_3X3
        """
        self.max3d_2a = nn.MaxPool3d((1, 3, 3),
                                     stride=(1, 2, 2),
                                     padding=(0, 1, 1))
        """
        Conv3d_2b_1x1
        """
        self.conv3d_2b = conv3d_bn(in_ch=64, out_ch=64)
        """
        Conv3d_2c_3x3
        """
        self.conv3d_2c = conv3d_bn(in_ch=64,
                                   out_ch=192,
                                   k=(3, 3, 3),
                                   p=(1, 1, 1))
        """
        MaxPool3d_2a_3X3
        """
        self.max3d_3a = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                     stride=(1, 2, 2),
                                     padding=(0, 1, 1))
        """
        Mixed_3b
        """
        self.Mixed_3b = Mixed(in_ch=192,
                              out_ch=(64, 96, 128, 16, 32,
                                      32))  # out_ch = 64+128+32+192=256
        """
        Mixed_3c
        """
        self.Mixed_3c = Mixed(in_ch=256, out_ch=(128, 128, 192, 32, 96,
                                                 64))  # out_ch = 480
        """
        Mixed 4a, 4b ,4c, 4d, 4e, 4f
        """
        self.max3d_4a = nn.MaxPool3d(kernel_size=(3, 3, 3),
                                     stride=(2, 2, 2),
                                     padding=(1, 1, 1))  # 1/8 size
        self.Mixed_4b = Mixed(in_ch=480, out_ch=(192, 96, 208, 16, 48,
                                                 64))  # out_ch = 512
        self.Mixed_4c = Mixed(in_ch=512, out_ch=(160, 112, 224, 24, 64,
                                                 64))  # out_ch = 512
        self.Mixed_4d = Mixed(in_ch=512, out_ch=(128, 128, 256, 24, 64,
                                                 64))  # out_ch = 512
        self.Mixed_4e = Mixed(in_ch=512, out_ch=(112, 144, 288, 32, 64,
                                                 64))  # out_ch = 528
        self.Mixed_4f = Mixed(in_ch=528, out_ch=(256, 160, 320, 32, 128,
                                                 128))  # out_ch = 832
        """
        MaxPool3d_5a_2x2
        """
        self.max3d_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        """
        Mixed 5b ,5c
        """
        self.Mixed_5b = Mixed(in_ch=832, out_ch=(256, 160, 320, 32, 128,
                                                 128))  # out_ch = 832
        self.Mixed_5c = Mixed(in_ch=832, out_ch=(384, 192, 384, 48, 128,
                                                 128))  # out_ch = 1024

    def forward(self, x):
        """
        init
        """
        outputs = dict()
        """
        Conv 1a
        """
        conv1a = self.conv3d_1a(x)
        outputs['Conv1a'] = conv1a
        """
        Max3d 2a
        Conv 2b, 2c
        """
        conv2c = self.conv3d_2c(self.conv3d_2b(self.max3d_2a(conv1a)))
        outputs['Conv2c'] = conv2c
        """
        Max3d 3a
        Mixed 3b, 3c
        """
        Mix3c = self.Mixed_3c(self.Mixed_3b(self.max3d_3a(conv2c)))
        outputs['Mix3c'] = Mix3c
        """
        Max3d 4a
        Mixed 4b, 4c, 4d, 4e, 4f
        """
        Mix4f = self.Mixed_4f(
            self.Mixed_4e(
                self.Mixed_4d(
                    self.Mixed_4c(self.Mixed_4b(self.max3d_4a(Mix3c))))))
        outputs['Mix4f'] = Mix4f
        """
        Max3d 5a,
        Mixed 5b, bc
        """
        Mix5c = self.Mixed_5c(self.Mixed_5b(self.max3d_5a(Mix4f)))
        outputs['Mix5c'] = Mix5c

        return outputs


if __name__ == '__main__':
    net = MLVR(3, 2).cuda()
    x = torch.randn(1, 3, 64, 224, 224).cuda()
    outputs = net(x)
