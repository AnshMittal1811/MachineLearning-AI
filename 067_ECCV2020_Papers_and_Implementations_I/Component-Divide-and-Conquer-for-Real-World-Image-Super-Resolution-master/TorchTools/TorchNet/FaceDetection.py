from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules import upsampleBlock
from .activation import swish


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4_1 = nn.Conv2d(32, 2, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1)

    def forward(self, input):
        conv3 = F.relu(self.conv3(
            F.relu(self.conv2(
                self.pool(F.relu(self.conv1(input)))
            ))
        ))
        conv4_1 = self.conv4_1(conv3)
        conv4_2 = self.conv4_2(conv3)
        return torch.cat([conv4_1, conv4_2], dim=1)


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 28, 3)
        self.pool1 = nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, 3)
        self.pool2 = nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, 2)
        self.linear1 = nn.Linear(576, 128)
        self.linear2_1 = nn.Linear(128, 2)
        self.linear2_2 = nn.Linear(128, 4)
        self.linear2_3 = nn.Linear(128, 10)

    def forward(self, input):
        conv3 = F.relu(self.conv3(
            self.pool2(F.relu(self.conv2(
                self.pool1(F.relu(self.conv1(input)))
            )))
        ))
        flat = conv3.view(conv3.size(0), -1)
        linear1 = F.relu(self.linear1(flat))
        classification = self.linear2_1()
        return None


def _detector_parser(image_size, ):
    pass


def _2detector_label():
    pass


class LightFaceDetector(nn.Module):
    """
    This is a lightweight face landmark detector using fully convolution layer. detect m points
    |-------------------------------|
    |     Input Y Channel Image     |
    |-------------------------------|
    |  Convolution_1 n32k5s1, ReLu  |
    |-------------------------------|
    |          MaxPooling 2x        |
    |-------------------------------|
    |  Convolution_2 n32k3s1, ReLu  |
    |-------------------------------|
    |  Convolution_3 n32k3s1, ReLu  |
    |-------------------------------|
    |        Pixel Shuffle 2x       |
    |-------------------------------|
    | Convolution_4 nmk3s1, Sigmoid |
    |-------------------------------|
    """
    def __init__(self, landmarks=5, activation=swish, in_channel=1):
        super(LightFaceDetector, self).__init__()
        self.act = activation
        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(in_channel, 32, 5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.upsample = upsampleBlock(32, 128)
        self.conv4 = nn.Conv2d(32, landmarks, 3, stride=1, padding=1)

    def forward(self, input):
        return F.sigmoid(self.conv4(self.upsample(self.act(self.conv3(self.act(self.conv2(self.pool(self.act(self.conv1(input))))))))))






