import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

from .activation import swish
from .modules import Flatten, upsampleBlock


class CalligraphyDiscriminator(nn.Module):
    def __init__(self, n_maps=64, activation=swish):
        super(CalligraphyDiscriminator, self).__init__()
        self.act = activation
        # B*1*32*32
        self.conv1 = nn.Conv2d(1, n_maps, kernel_size=3, stride=2, padding=1)
        # B*N*16*16
        self.conv2 = nn.Conv2d(n_maps*1, n_maps*2, kernel_size=3, stride=2, padding=1)
        # B*2N*8*8
        self.conv3 = nn.Conv2d(n_maps*2, n_maps*4, kernel_size=3, stride=2, padding=1)
        # B*4N*4*4
        self.conv4 = nn.Conv2d(n_maps*4, n_maps*8, kernel_size=3, stride=2, padding=1)
        # B*8N*2*2
        self.flat = Flatten()
        self.dense = nn.Linear(2048, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, input):
        return F.sigmoid(
            self.out(
                self.act(self.dense(
                    self.flat(
                        self.act(self.conv4(
                            self.act(self.conv3(
                                self.act(self.conv2(
                                    self.act(self.conv1(input))
                                ))
                            ))
                        ))
                    )
                ))
            )
        )


class CalligraphyUpsampleDiscriminator(nn.Module):
    def __init__(self, n_maps=32, activation=swish):
        super(CalligraphyUpsampleDiscriminator, self).__init__()
        self.act = activation
        # B*1*128*128
        self.conv1 = nn.Conv2d(1, n_maps, kernel_size=5, stride=2, padding=2)
        # B*N*64*64
        self.conv2 = nn.Conv2d(n_maps*1, n_maps*1, kernel_size=5, stride=2, padding=2)
        # B*N*32*32
        self.conv3 = nn.Conv2d(n_maps*1, n_maps*2, kernel_size=3, stride=2, padding=1)
        # B*2N*16*16
        self.conv4 = nn.Conv2d(n_maps*2, n_maps*4, kernel_size=3, stride=2, padding=1)
        # B*4N*8*8
        self.conv5 = nn.Conv2d(n_maps*4, n_maps*4, kernel_size=3, stride=2, padding=1)
        # B*4N*4*4
        self.conv6 = nn.Conv2d(n_maps*4, n_maps*8, kernel_size=3, stride=2, padding=1)
        # B*8N*2*2
        self.flat = Flatten()
        self.dense = nn.Linear(2048, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, input):
        return F.sigmoid(
            self.out(
                self.act(self.dense(
                    self.flat(self.act(self.conv6(
                        self.act(self.conv5(
                            self.act(self.conv4(
                                self.act(self.conv3(
                                    self.act(self.conv2(
                                        self.act(self.conv1(input))
                                    ))
                                ))
                            ))
                        ))
                    )))
                ))
            )
        )


class CalliUpsampleNet(nn.Module):
    def __init__(self, n_maps=64, activation=swish):
        super(CalliUpsampleNet, self).__init__()
        self.act = activation


class EDCalliTransferNet(nn.Module):
    """
    Encoder-Decoder Transfer Net
    Use Pixel-shuffle as Upsample method
    Input 32*32
    """
    def __init__(self, n_maps=64, activation=swish):
        super(EDCalliTransferNet, self).__init__()
        self.act = activation
        self.flat = Flatten()
        # B*1*32*32
        self.E_Conv = nn.Conv2d(in_channels=1, out_channels=n_maps*1, kernel_size=5, stride=1, padding=2)
        # B*N*32*32
        self.E_conv1 = nn.Conv2d(in_channels=n_maps*1, out_channels=n_maps*1, kernel_size=3, stride=2, padding=1)
        # B*N*16*16
        self.E_conv2 = nn.Conv2d(in_channels=n_maps*1, out_channels=n_maps*2, kernel_size=3, stride=2, padding=1)
        # B*2N*8*8
        self.E_conv3 = nn.Conv2d(in_channels=n_maps*2, out_channels=n_maps*4, kernel_size=3, stride=2, padding=1)
        # B*4N*4*4
        self.E_conv4 = nn.Conv2d(in_channels=n_maps*4, out_channels=n_maps*8, kernel_size=3, stride=2, padding=1)
        # B*8N*2*2
        self.E_conv5 = nn.Conv2d(in_channels=n_maps*8, out_channels=n_maps*16, kernel_size=2)
        # B*16N*1*1
        # if N == 64, 16*N == 1024
        self.pixel = nn.PixelShuffle(2)
        # pixel
        # B*4N*2*2
        self.D_conv1 = nn.Conv2d(in_channels=n_maps*4, out_channels=n_maps*8, kernel_size=3, stride=1, padding=1)
        # B*8N*2*2
        # pixel
        # B*2N*4*4
        self.D_conv2 = nn.Conv2d(in_channels=n_maps*2, out_channels=n_maps*4, kernel_size=3, stride=1, padding=1)
        # B*4N*4*4
        # pixel
        # B*N*8*8
        self.D_conv3 = nn.Conv2d(in_channels=n_maps*1, out_channels=n_maps*4, kernel_size=5, stride=1, padding=2)
        # B*4N*8*8
        # pixel
        # B*N*16*16
        self.D_conv4 = nn.Conv2d(in_channels=n_maps*1, out_channels=n_maps*4, kernel_size=5, stride=1, padding=2)
        # B*4N*16*16
        # pixel
        # B*N*32*32
        self.D_conv5 = nn.Conv2d(in_channels=n_maps*1, out_channels=n_maps, kernel_size=3, stride=1, padding=1)
        # B*N*32*32
        self.D_Conv = nn.Conv2d(in_channels=n_maps, out_channels=1, kernel_size=3, stride=1, padding=1)
        # B*1*32*32

    def _decoder(self, code):
        return F.tanh(
            self.D_Conv(
                self.act(self.D_conv5(
                    self.pixel(self.act(self.D_conv4(
                        self.pixel(self.act(self.D_conv3(
                            self.pixel(self.act(self.D_conv2(
                                self.pixel(self.act(self.D_conv1(
                                    self.pixel(code)
                                )))
                            )))
                        )))
                    )))
                ))
            )
        )

    def _enocder(self, input):
        """
        :param input: 32*32
        :return: B*16N*1*1
        """
        return self.act(self.E_conv5(
            self.act(self.E_conv4(
                    self.act(self.E_conv3(
                        self.act(self.E_conv2(
                            self.act(self.E_conv1(
                                self.act(self.E_Conv(input))
                            ))
                        ))
                    ))
                ))
            ))

    def forward(self, input):
        code = self._enocder(input)
        return self._decoder(code)


class CaliNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(CaliNet, self).__init__()
        self.conv = nn.Sequential(
            *[
                nn.Conv2d(input_nc, ngf, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(ngf, ngf * 2, kernel_size=5, stride=2, padding=2),
                norm_layer(ngf * 2),
                nn.ReLU(),
                nn.Conv2d(ngf * 2, ngf * 2, kernel_size=5, stride=1, padding=2),
                norm_layer(ngf * 2),
                nn.ReLU(),
                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=5, stride=2, padding=2),
                norm_layer(ngf * 4),
                nn.ReLU(),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=5, stride=1, padding=2),
                norm_layer(ngf * 4),
                nn.ReLU(),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=5, stride=2, padding=2),
                norm_layer(ngf * 4),
                nn.ReLU(),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=5, stride=1, padding=2),
                norm_layer(ngf * 4),
                nn.ReLU(),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=5, stride=1, padding=2),
                norm_layer(ngf * 4),
                nn.ReLU(),
            ]
        )
        self.decv = nn.Sequential(
            *[
                nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=4, stride=2, padding=1),
                norm_layer(ngf * 4),
                nn.ReLU(),
                nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=5, stride=1, padding=2),
                norm_layer(ngf * 4),
                nn.ReLU(),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
                norm_layer(ngf * 2),
                nn.ReLU(),
                nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=5, stride=1, padding=2),
                norm_layer(ngf * 2),
                nn.ReLU(),
                nn.ConvTranspose2d(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(ngf * 1, output_nc, kernel_size=5, stride=1, padding=2)
            ]
        )

    def forward(self, input):
        x = self.conv(input)
        x = self.decv(x)
        return x


