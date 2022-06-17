import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
        residual=False
    ):
        super().__init__()
        self.residual = residual
        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )
        if in_channel!=out_channel:
            self.ch_match = EqualConv2d(in_channel,out_channel,1,1,0)
        else:
            self.ch_match = nn.Identity()
        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        res = self.ch_match(input)
        
        out = self.conv1(input)
        out = self.conv2(out)
        if self.residual:
            return out + res
        else:
            return out


class AdaptiveAttention(nn.Module):
    def __init__(self, img_dim,style_dim):
        super().__init__()

        self.img_dim = img_dim
        self.fc = EqualLinear(style_dim, (img_dim**2))
        self.gamma = nn.Parameter(torch.ones(1,1,1,1))

    def forward(self, x, p):

        h = self.fc(p)
        h = h.view(h.size(0), 1, self.img_dim, self.img_dim)
        h = F.sigmoid(h)

        return self.gamma*(h*x)+x
        
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(chunks=2, dim=1)
        
        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        img_dim=0,
        initial=False,
        upsample=False,
        fused=False,
        use_att=True
    ):
        super().__init__()
        self.use_att = use_att
        if initial:
            self.conv1 = ConstantInput(in_channel)
        
        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )
                
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        if use_att:
            self.adaat1 = AdaptiveAttention(img_dim,style_dim)
            self.adaat2 = AdaptiveAttention(img_dim,style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, pix,style2=None,pix2=None,eval_mode=False):
        
        if eval_mode == False:
            pix2 = pix
            style2 = style
        
        out = self.conv1(input)
        if self.use_att:
            out = self.adaat1(out, pix)
        out = self.adain1(out, style)
        out = self.lrelu1(out)

        out = self.conv2(out)
        if self.use_att:
            out = self.adaat2(out,pix2)
        out = self.adain2(out, style2)
        out = self.lrelu2(out)

        return out

class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
#         style_dim=512,
        img_dim=0,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.in1 = nn.InstanceNorm2d(out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.in2 = nn.InstanceNorm2d(out_channel)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, pix):
        out = self.conv1(input)
        out = self.in1(out)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.lrelu2(out)

        return out

class Generator(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, img_dim = 4,initial=True),  # 4
                StyledConvBlock(512, 512, 3, 1, img_dim = 8,upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, img_dim = 16,upsample=True),  # 16
                StyledConvBlock(512, 512, 3, 1, img_dim = 32,upsample=True),  # 32
                StyledConvBlock(512, 256, 3, 1, img_dim = 64,upsample=True),  # 64
                StyledConvBlock(256, 128, 3, 1, img_dim = 128,upsample=True),  # 128
                StyledConvBlock(128, 64, 3, 1, img_dim = 256,upsample=True),  # 256
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),

            ]
        )

        # self.blur = Blur()

    def forward(self, style, pix, step=0, alpha=-1,eval_mode=False):
        if eval_mode:
            out = style[0]
        else:
            out = style
        
        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if i > 0 and step > 0:
                out_prev = out
            if eval_mode:
                out = conv(out,style[2*i],pix[2*i],style[2*i+1],pix[2*i+1],eval_mode=eval_mode)
            else:
                out = conv(out,style,pix)
            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out

class MappingNets(nn.Module):
    def __init__(self,code_dim=512,n_mlp = 8,num_domains=2):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_mlp//2):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style_shared = nn.Sequential(*layers)
        layers2 = [PixelNorm()]
        for i in range(n_mlp):
            layers2.append(EqualLinear(code_dim, code_dim))
            layers2.append(nn.LeakyReLU(0.2))
        self.pix_shared = nn.Sequential(*layers2)
        
        self.style_unshared = nn.ModuleList()
        
        for _ in range(num_domains):
            self.style_unshared.append(nn.Sequential(EqualLinear(code_dim,code_dim),
                                                    nn.LeakyReLU(0.2),
                                                    EqualLinear(code_dim,code_dim),
                                                    nn.LeakyReLU(0.2),
                                                    EqualLinear(code_dim,code_dim),
                                                    nn.LeakyReLU(0.2),
                                                    EqualLinear(code_dim,code_dim),
                                                     nn.LeakyReLU(0.2)
                                                    ))

        
    def forward(self,input1,input2,y):
        style = self.style_shared(input1)
        pix = self.pix_shared(input2)
        
        styles = []
        for layer in self.style_unshared:
            styles += [layer(style)]
        styles = torch.stack(styles, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        styles = styles[idx, y]

        return styles,pix
class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512,num_domains=2, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)
        self.mapping = MappingNets(code_dim,n_mlp,num_domains)

    def forward(
        self,
        input,
        input2,
        y,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        mean_pix=None,
        fix_style = False,
        fix_pix = False,
        style_weight=0,
        pix_weight=0,
        eval_mode=False
        
    ):

        styles,pix = self.mapping(input,input2,y)
        

        batch = input.shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
#                 noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            mean_style = mean_style.repeat(batch,1)
            styles_norm = mean_style + style_weight * (styles - mean_style)
            
            styles = styles_norm
            if mixing_range!=(-1,-1):
                style_norm2 = mean_style + style_weight * (styles2 - mean_style)
                styles2 = style_norm2
            if fix_style:
                styles = mean_style
        if mean_pix is not None:
            mean_pix = mean_pix.repeat(batch,1)
            pix_norm = mean_pix + pix_weight*(pix-mean_pix)
            
            pix = pix_norm

            if fix_pix: 
                pix = mean_pix

        return self.generator(styles, pix, step, alpha,eval_mode)

    def mean_style(self, input,input2):
        style = self.style(input).mean(0, keepdim=True)
        pix = self.pix(input2).mean(0,keepdim=True)
        return style,pix


class Discriminator(nn.Module):
    def __init__(self, fused=True, num_domains=2,from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [

                ConvBlock(64, 128, 3, 1, downsample=True),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [

                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, num_domains)

    def forward(self, input, y,step=0, alpha=-1):
        feats = []
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)
            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]
        
        return out

class StyleEncoder(nn.Module):
    def __init__(self, fused=True, num_domains=2,style_dim=512,from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(64, 128, 3, 1, downsample=False,residual=True),  # 128
                ConvBlock(128, 256, 3, 1, downsample=False,residual=True),  # 64
                ConvBlock(256, 512, 3, 1, downsample=False,residual=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=False,residual=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=False,residual=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=False,residual=True),  # 4
                ConvBlock(512, 512, 3, 1, 4, 0, downsample=False),
            ]
        )
        self.avgpool = nn.AvgPool2d(2)
        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )
        
        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [EqualLinear(512, style_dim)]

    def forward(self, input,y, step=0, alpha=-1):
        feats = []
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            out = self.progression[index](out)
            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out
            if i != 0:
                out = self.avgpool(out)
        
        out = out.view(out.size(0), -1)
        sty = []
        for layer in self.unshared:
            sty += [layer(out)]
        sty = torch.stack(sty, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = sty[idx, y]  # (batch, style_dim)
        
        return s

class ContentEncoder(nn.Module):
    def __init__(self, fused=True,style_dim=512,from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [

                ConvBlock(64, 128, 3, 1, downsample=False,residual=True),  # 128
                ConvBlock(128, 256, 3, 1, downsample=False,residual=True),  # 64
                ConvBlock(256, 512, 3, 1, downsample=False,residual=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=False,residual=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=False,residual=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=False,residual=True),  # 4
                ConvBlock(512, 512, 3, 1, 4, 0, downsample=False),
            ]
        )
        self.avgpool = nn.AvgPool2d(2)
        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [

                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )
        
        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512,style_dim)

    def forward(self, input, step=0, alpha=-1):
        feats = []
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            out = self.progression[index](out)
    
            if i > 0:
                
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out
            if i != 0:
                out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        c = self.linear(out)
        return c