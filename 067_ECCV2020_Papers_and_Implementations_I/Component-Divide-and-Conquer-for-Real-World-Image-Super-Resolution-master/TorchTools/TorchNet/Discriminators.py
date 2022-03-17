import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import swish


class Discriminator(nn.Module):

    def __init__(self, activation=swish):
        super(Discriminator, self).__init__()
        self.act = activation

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.act(self.conv1(x))

        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        x = self.act(self.bn6(self.conv6(x)))
        x = self.act(self.bn7(self.conv7(x)))
        x = self.act(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


class DiscriminatorY(nn.Module):

    def __init__(self, activation=swish):
        super(DiscriminatorY, self).__init__()
        self.act = activation
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.act(self.conv1(x))

        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        x = self.act(self.bn6(self.conv6(x)))
        x = self.act(self.bn7(self.conv7(x)))
        x = self.act(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

