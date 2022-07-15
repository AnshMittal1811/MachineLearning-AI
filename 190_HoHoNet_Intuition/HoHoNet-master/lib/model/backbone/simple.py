import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleResBlock(nn.Module):
    def __init__(self, a, b, c, s):
        super(SimpleResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(a, b, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, stride=s, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(a, c, 1, stride=s, bias=False),
            nn.BatchNorm2d(c),
        )
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.layer[-1].weight, 0)
        nn.init.constant_(self.layer[-1].bias, 0)

    def forward(self, x):
        return self.relu(self.layer(x) + self.skip(x))

class SimpleConv3x3Block(nn.Module):
    def __init__(self, a, b, c, s):
        super(SimpleConv3x3Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(a, c, 3, padding=1, stride=s, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

def SimpleConv3x3MaxBlock(a, b, c, s):
    return nn.Sequential(
        nn.Conv2d(a, c, 3, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c, 3, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(s, stride=s),
    )

def SimpleConv3x3lBlock(a, b, c, s):
    return nn.Sequential(
        nn.Conv2d(a, c, 3, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c, 3, padding=1, stride=s, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
    )


class SimpleEncoder(nn.Module):
    def __init__(self, input_extra=0, input_height=512, block='res', expand=1):
        super(SimpleEncoder, self).__init__()
        self.conv_pre = nn.Sequential(
            nn.Conv2d(3+input_extra, 16*expand, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16*expand),
            nn.ReLU(inplace=True),
        )

        if block == 'res':
            Block = SimpleResBlock
        elif block == 'conv3x3':
            Block = SimpleConv3x3Block
        elif block == 'conv3x3l':
            Block = SimpleConv3x3lBlock
        elif block == 'conv3x3max':
            Block = SimpleConv3x3MaxBlock
        else:
            raise NotImplementedError
        self.block0 = Block(16*expand, 16*expand, 32*expand, 2)
        self.block1 = Block(32*expand, 32*expand, 64*expand, 2)
        self.block2 = Block(64*expand, 64*expand, 128*expand, 2)
        self.block3 = Block(128*expand, 128*expand, 256*expand, 2)
        self.block4 = Block(256*expand, 256*expand, 256*expand, 2)

        self.out_channels = [64*expand, 128*expand, 256*expand, 256*expand]
        self.feat_heights = [input_height//4//(2**i) for i in range(4)]

    def forward(self, x):
        features = []
        x = self.conv_pre(x)
        x = self.block0(x)
        x = self.block1(x);  features.append(x)  # 1/4
        x = self.block2(x);  features.append(x)  # 1/8
        x = self.block3(x);  features.append(x)  # 1/16
        x = self.block4(x);  features.append(x)  # 1/32
        return features
