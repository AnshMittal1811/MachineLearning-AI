import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu
from torch.autograd import Variable

from .activation import swish

def GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=-1.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(np.random.normal(loc=mean, scale=sigma, size=size))
    return torch.clamp(noise + tensor, min=min, max=max)


def PoissonNoising(tensor, lamb, noise_size=None, min=-1.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(np.random.poisson(lam=lamb, size=size))
    return torch.clamp(noise + tensor, min=min, max=max)


class Blur(nn.Module):
    def __init__(self, l=15):
        super(Blur, self).__init__()
        self.l = l
        self.pad = nn.ReflectionPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        if C != 1:
            raise Exception('Plz use Y channel')
        kernel_var = Variable(kernel.view((1, C, self.l, self.l)))
        return F.conv2d(self.pad(input), kernel_var)


class BatchBlur(nn.Module):
    def __init__(self, l=15):
    # def __init__(self, l=15):
        super(BatchBlur, self).__init__()
        self.l = l
        # self.pad = nn.ReflectionPad2d(l // 2)
        self.pad = None

    def forward(self, input, kernel):
        self.pad = nn.ReflectionPad2d(kernel.shape[-1] // 2)
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]
        input_CBHW = pad.view((C, B, H_p, W_p))
        if C != 1:
            raise Exception('Plz use Y channel')
        # kernel_var = kernel.view((B, C, self.l, self.l))
        kernel_var = kernel.view((B, C, kernel.shape[-2], kernel.shape[-1]))

        return F.conv2d(input_CBHW, kernel_var, groups=B).view((B, C, H, W))


class GaussianBlur(nn.Module):
    def __init__(self, input_channel, l=3, sigma=0.6):
        super(GaussianBlur, self).__init__()
        self.l = l
        self.sig = sigma
        self.kernel = Variable(self._g_kernel().view((1, input_channel, l, l)))
        self.pad = nn.ReflectionPad2d(l // 2)

    def _g_kernel(self):
        ax = np.arange(-self.l // 2 + 1., self.l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * self.sig ** 2))
        return torch.FloatTensor(kernel / np.sum(kernel))

    def forward(self, input):
        return F.conv2d(self.pad(input), self.kernel)


class RandomBlur(nn.Module):
    def __init__(self, input_channel=1, kernel_size=15, sigma_min=0.0, sigma_max=1.4):
        super(RandomBlur, self).__init__()
        self.min = sigma_min
        self.max = sigma_max
        self.l = kernel_size
        self.input_channel = input_channel
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def _g_kernel(self, sigma):
        ax = np.arange(-self.l // 2 + 1., self.l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        return torch.FloatTensor(kernel / np.sum(kernel))

    def forward(self, input):
        sigma_random = random.uniform(self.min, self.max)
        kernel = Variable(self._g_kernel(sigma_random).view((1, self.input_channel, self.l, self.l)))
        return F.conv2d(self.pad(input), kernel)


class RandomNoisedBlur(nn.Module):
    def __init__(self, input_channel=1, kernel_size=15, sigma_min=0.0, sigma_max=1.4, noise_sigma=0.02):
        super(RandomNoisedBlur, self).__init__()
        self.min = sigma_min
        self.max = sigma_max
        self.l = kernel_size
        self.noise = noise_sigma
        self.input_channel = input_channel
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def _g_kernel(self, sigma):
        ax = np.arange(-self.l // 2 + 1., self.l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        noise = np.random.normal(1.0, self.noise, kernel.shape)
        gaussian = kernel / np.sum(kernel)
        return torch.FloatTensor(gaussian * noise)

    def forward(self, input):
        sigma_random = random.uniform(self.min, self.max)
        kernel = Variable(self._g_kernel(sigma_random).view((1, self.input_channel, self.l, self.l)))
        return F.conv2d(self.pad(input), kernel)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class FeatureExtractor(nn.Module):

    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        # TODO convert x: RGB to BGR
        return self.features(x)


class residualBlock(nn.Module):

    def __init__(self, in_channels=64, kernel=3, mid_channels=64, out_channels=64, stride=1, activation=relu):
        super(residualBlock, self).__init__()
        self.act = activation
        self.pad1 = nn.ReflectionPad2d((kernel // 2))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.pad2 = nn.ReflectionPad2d((kernel // 2))
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(self.pad1(x))))
        return self.bn2(self.conv2(self.pad2(y))) + x


class residualBlockNoBN(nn.Module):

    def __init__(self, in_channels=64, kernel=3, mid_channels=64, out_channels=64, stride=1, activation=relu):
        super(residualBlockNoBN, self).__init__()
        self.act = activation
        self.pad1 = nn.ReflectionPad2d((kernel // 2))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel, stride=stride, padding=0)
        self.pad2 = nn.ReflectionPad2d((kernel // 2))
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=0)

    def forward(self, x):
        y = self.act(self.conv1(self.pad1(x)))
        return self.conv2(self.pad2(y)) + x


class residualBlockIN(nn.Module):
    def __init__(self, in_channels=64, kernel=3, mid_channels=64, out_channels=64, stride=1, activation=relu):
        super(residualBlockIN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.in1 = nn.InstanceNorm2d(mid_channels, affine=True)
        self.act = activation
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.act(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class upsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation=relu):
        super(upsampleBlock, self).__init__()
        self.act = activation
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=0)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return self.act(self.shuffler(self.conv(self.pad(x))))


class deconvUpsampleBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_1=5, kernel_2=3, activation=relu):
        self.act = activation
        super(deconvUpsampleBlock, self).__init__()
        self.deconv_1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_1, stride=2, padding=kernel_1 // 2)
        # self.deconv_2 = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=kernel_2, padding=kernel_2 // 2)

    def forward(self, x):
        return self.act(self.deconv_1(x))


class Features4Layer(nn.Module):
    """
    Basic feature extractor, 4 layer version
    """
    def __init__(self, features=64, activation=relu):
        """
        :param frame: The input frame image
        :param features: feature maps per layer
        """
        super(Features4Layer, self).__init__()
        self.act = activation

        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(1, features, 5, stride=1, padding=0)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(features)

        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(features, features, 3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(features)

        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(features, features, 3, stride=1, padding=0)

    def forward(self, frame):
        return self.act(self.conv4(self.pad4(
            self.act(self.bn3(self.conv3(self.pad3(
                self.act(self.bn2(self.conv2(self.pad2(
                    self.act(self.conv1(self.pad1(frame)))
                ))))
            ))))
        )))


class Features3Layer(nn.Module):
    """
    Basic feature extractor, 4 layer version
    """
    def __init__(self, features=64, activation=relu):
        """
        :param frame: The input frame image
        :param features: feature maps per layer
        """
        super(Features3Layer, self).__init__()
        self.act = activation

        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(1, features, 5, stride=1, padding=0)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(features)

        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(features, features, 3, stride=1, padding=0)

    def forward(self, frame):
        return self.act(self.conv3(self.pad3(
            self.act(self.bn2(self.conv2(self.pad2(
                self.act(self.conv1(self.pad1(frame)))
            ))))
        )))


class LateUpsamplingBlock(nn.Module):
    """
    this is another up-sample block for step upsample
    |------------------------------|
    |           features           |
    |------------------------------|
    |   n   |   residual blocks    |
    |------------------------------|
    | Pixel shuffle up-sampling x2 |
    |------------------------------|
    """
    def __init__(self, features=64, n_res_block=3):
        """
        :param features: number of feature maps input
        :param n_res_block: number of residual blocks
        """
        super(LateUpsamplingBlock, self).__init__()
        self.n_residual_blocks = n_res_block

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock(features))

        self.upsample = upsampleBlock(features, features * 4)

    def forward(self, features):
        for i in range(self.n_residual_blocks):
            features = self.__getattr__('residual_block' + str(i + 1))(features)
        return self.upsample(features)


class LateUpsamplingBlockNoBN(nn.Module):
    """
    this is another up-sample block for step upsample
    |------------------------------|
    |           features           |
    |------------------------------|
    |   n   |   residual blocks    |
    |------------------------------|
    | Pixel shuffle up-sampling x2 |
    |------------------------------|
    """
    def __init__(self, features=64, n_res_block=3):
        """
        :param features: number of feature maps input
        :param n_res_block: number of residual blocks
        """
        super(LateUpsamplingBlockNoBN, self).__init__()
        self.n_residual_blocks = n_res_block

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlockNoBN(features))

        self.upsample = upsampleBlock(features, features * 4)

    def forward(self, features):
        for i in range(self.n_residual_blocks):
            features = self.__getattr__('residual_block' + str(i + 1))(features)
        return self.upsample(features)


class DownsamplingShuffle(nn.Module):

    def __init__(self, scala):
        super(DownsamplingShuffle, self).__init__()
        self.scala = scala

    def forward(self, input):
        """
        input should be 4D tensor N, C, H, W
        :param input:
        :return:
        """
        N, C, H, W = input.size()
        assert H % self.scala == 0, 'Plz Check input and scala'
        assert W % self.scala == 0, 'Plz Check input and scala'
        map_channels = self.scala ** 2
        channels = C * map_channels
        out_height = H // self.scala
        out_width = W // self.scala

        input_view = input.contiguous().view(
            N, C, out_height, self.scala, out_width, self.scala)

        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

        return shuffle_out.view(N, channels, out_height, out_width)


class _AttentionDownConv(nn.Module):
    def __init__(self, features=16):
        super(_AttentionDownConv, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        return F.relu(self.downsample(
            F.relu(self.conv2(
                F.relu(self.conv1(input))
            ))
        ))


class _AttentionUpConv(nn.Module):
    def __init__(self, features=16):
        super(_AttentionUpConv, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):
        return F.relu(self.upsample(
            F.relu(self.conv2(
                F.relu(self.conv1(input))
            ))
        ))


class Attention(nn.Module):
    """
    Attention Module, output with sigmoid
    """
    def __init__(self, input_channel=1, feature_channels=16, down_samples=2):
        super(Attention, self).__init__()
        self.input = input_channel
        self.ngf = feature_channels
        self.down = down_samples
        self.down_square = 2 ** down_samples

        self.input_conv = nn.Conv2d(input_channel, feature_channels, kernel_size=5, stride=1, padding=2)

        self.final_conv = nn.Conv2d(feature_channels, 1, kernel_size=5, stride=1, padding=2)

        for i in range(down_samples):
            self.add_module('down_sample_' + str(i + 1), _AttentionDownConv(features=feature_channels))

        for i in range(down_samples):
            self.add_module('up_sample_' + str(i + 1), _AttentionUpConv(features=feature_channels))

    def forward(self, input):
        B, C, H, W = input.size()
        pad_H = self.down_square - (H % self.down_square) if H % self.down_square != 0 else 0
        pad_W = self.down_square - (W % self.down_square) if W % self.down_square != 0 else 0

        input_pad = F.pad(input, (0, pad_H, 0, pad_W), 'reflect')


        output = F.relu(self.input_conv(input_pad))

        for i in range(self.down):
            output = self.__getattr__('down_sample_' + str(i + 1))(output)

        for i in range(self.down):
            output = self.__getattr__('up_sample_' + str(i + 1))(output)

        output = self.final_conv(output)
        output_pad = F.pad(output, (0, -pad_H, 0, -pad_W))

        return F.sigmoid(output_pad)


class BatchBicubicDownsampler(nn.Module):
    """
    batch downsample using bicubic interpolation
    """
    def __init__(self, factor, use_cuda=False):
        super(BatchBicubicDownsampler, self).__init__()
        self.scale_factor = factor
        self.use_cuda = use_cuda

        filter_height = factor * 4
        filter_width = factor * 4
        strides = factor
        pad_along_height = max(filter_height - strides, 0)
        pad_along_width = max(filter_width - strides, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        def build_filter(factor):
            size = factor * 4
            k = np.zeros((size))
            for i in range(size):
                x = (1 / factor) * (i - np.floor(size / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            k = np.outer(k, k.T)
            k = torch.FloatTensor(k).view(1, 1, *k.shape)
            return k

        self.kernel = Variable(build_filter(self.scale_factor))
        self.pad = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        if use_cuda:
            self.kernel = self.kernel.cuda()

    def forward(self, x):
        x = self.pad(x)
        x = nn.functional.conv2d(x, self.kernel, stride=self.scale_factor)
        return x


class BatchBicubicUpsampler(nn.Module):
    def __init__(self, factor, use_cuda=False):
        super(BatchBicubicUpsampler, self).__init__()
        self.factor = factor
        def R(n):
            a = -0.5
            n = abs(n)
            if n <= 1:
                return (a + 2) * (n ** 3) - (a + 3) * (n ** 2) + 1
            elif n > 1 and n < 2:
                return a * (n ** 3) - 5 * a * (n ** 2) + 8 * a * n - 4 * a

            else:
                return 0

        kernel = torch.zeros(factor**2, 4, 4)
        cnt = 0
        xrange = np.arange(0, 1, 1 / factor)
        yrange = np.arange(0, 1, 1 / factor)

        for dx in xrange:
            for dy in yrange:
                for i in range(4):
                    for j in range(4):
                        kernel[cnt, i, j] = R(i - 1 - dx) * R(dy - j + 1)
                cnt += 1

        for i in range(4):
            kernel[i] /= torch.sum(kernel[i])

        self.kernel = Variable(torch.unsqueeze(kernel, 1))
        if use_cuda:
            self.kernel = self.kernel.cuda()

    def forward(self, input):
        input = nn.functional.pad(input, (1, 2, 1, 2), mode='replicate')
        hr_batch = nn.functional.conv2d(input, self.kernel)
        hr = nn.functional.pixel_shuffle(hr_batch, self.factor)
        hr = torch.clamp(hr, min=0, max=1.0)

        return hr
