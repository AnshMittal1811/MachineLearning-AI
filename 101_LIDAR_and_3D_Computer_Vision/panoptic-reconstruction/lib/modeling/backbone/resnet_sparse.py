from torch import nn
import MinkowskiEngine as Me


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = Me.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = Me.MinkowskiInstanceNorm(planes)
        self.conv2 = Me.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = Me.MinkowskiInstanceNorm(planes)
        self.relu = Me.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = Me.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = Me.MinkowskiInstanceNorm(planes)

        self.conv2 = Me.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = Me.MinkowskiInstanceNorm(planes)

        self.conv3 = Me.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = Me.MinkowskiInstanceNorm(
            planes * self.expansion)

        self.relu = Me.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
