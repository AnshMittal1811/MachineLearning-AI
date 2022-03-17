import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

from .activation import swish
from .modules import residualBlock, upsampleBlock


class Generator(nn.Module):
    """
    General Generator with pixel shuffle upsample
    """
    def __init__(self, n_residual_blocks, scala, activation=swish):
        """
        :param n_residual_blocks: Number of residual blocks
        :param scala: factor of upsample
        :param activation: function of activation
        """
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.scala = scala
        self.act = activation

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x = self.act(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return F.tanh(self.conv3(x))


class GeneratorY(nn.Module):
    """
    General Generator with pixel shuffle upsample
    """
    def __init__(self, n_residual_blocks, scala, activation=swish):
        """
        :param n_residual_blocks: Number of residual blocks
        :param scala: factor of upsample
        :param activation: function of activation
        """
        super(GeneratorY, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.scala = scala
        self.act = activation

        self.conv1 = nn.Conv2d(1, 64, 5, stride=1, padding=2)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 1, 7, stride=1, padding=3)

    def forward(self, x):
        x = self.act(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return F.tanh(self.conv3(x))


class VideoSRGAN_5(nn.Module):

    def __init__(self, n_residual_blocks, scala, activation=swish):
        super(VideoSRGAN_5, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.scala = scala
        self.time_window = 5
        self.act = activation

        self.conv_1_center = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.conv_1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.conv_1_2 = nn.Conv2d(1, 32, 5, stride=1, padding=2)

        self.shrink_conv = nn.Conv2d(32 * 5, 64, 1, stride=1, padding=0)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(int(log2(self.scala))):
            self.add_module('upsample' + str(i + 1), upsampleBlock(64, 256))

        self.final_pad = nn.ReflectionPad2d(3)
        self.conv3 = nn.Conv2d(64, 1, 7, stride=1, padding=0)

    def forward(self, frame_1, frame_2, frame_center, frame_4, frame_5):
        center_conv = self.conv_1_center(frame_center)
        conv_frame_1 = self.conv_1_2(frame_1)
        conv_frame_2 = self.conv_1_1(frame_2)
        conv_frame_4 = self.conv_1_1(frame_4)
        conv_frame_5 = self.conv_1_2(frame_5)
        conv_1 = self.act(torch.cat((conv_frame_1, conv_frame_2, center_conv, conv_frame_4, conv_frame_5), dim=1))
        x = self.act(self.shrink_conv(conv_1))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(log2(self.scala))):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return F.tanh(self.conv3(self.final_pad(x)))


