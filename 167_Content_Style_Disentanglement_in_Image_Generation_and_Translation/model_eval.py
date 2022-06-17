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
    ):
        super().__init__()

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
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveAttention(nn.Module):
    def __init__(self, img_dim,style_dim):
        super().__init__()

        self.img_dim = img_dim
        self.fc = EqualLinear(style_dim, (img_dim**2))
        self.gamma = nn.Parameter(torch.ones(1,1,1,1))

    def forward(self, x, p,loc=None,add_val=False,return_att = False,sub=False):

        h = self.fc(p)
        h = h.view(h.size(0), 1, self.img_dim, self.img_dim)
        h = F.sigmoid(h)
        if loc is not None:
            if add_val:
                h = loc + h
            elif sub:
                h = loc
            else:
                h = loc*h

        if return_att:
            return self.gamma*(h*x)+x, h 
        else:
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
        use_att=True,
        use_style=True
    ):
        super().__init__()
        self.use_att = use_att
        self.use_style = use_style
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
        if use_style==False:
            self.in1 = nn.InstanceNorm2d(out_channel)
            self.in2 = nn.InstanceNorm2d(out_channel)

        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)

        if use_att:
            self.adaat1 = AdaptiveAttention(img_dim,style_dim)
            self.adaat2 = AdaptiveAttention(img_dim,style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style,style2, pix,pix2):

        out = self.conv1(input)
        
        if self.use_att:
            out= self.adaat1(out,pix)
        if self.use_style:
            out = self.adain1(out, style)
        else:
            out = self.in1(out)
        out = self.lrelu1(out)
        
        

        out = self.conv2(out)

        if self.use_att:
            out= self.adaat2(out,pix2)
        if self.use_style:
            out = self.adain2(out, style2)
        else:
            out = self.in2(out)
        out = self.lrelu2(out)
        
        if self.use_att:
            if return_att:
                return out,a1,a2
            else:
                return out
        else:
            return out

class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
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

    def forward(self, input, style, pix,use_ppl=False,eps=False,rand_t = None):
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
                StyledConvBlock(64, 32, 3, 1, img_dim = 512,upsample=True,fused=True,use_att=False),  # 512
                StyledConvBlock(32, 16, 3, 1, img_dim = 1024,upsample=True,fused=True,use_att=False),  # 1024
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
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

        # self.blur = Blur()

    def forward(self, styles, pixs, step=0, alpha=-1, style2=None,pix2=None, mixing_range=(-1, -1),use_ppl=False,eps=False,rand_t = None,bin_idx=[0,0],loc=None,loc_idx = None,return_att=False,add_val=False,loc_id = [0,0],sub=False):
        out = styles[0]
        atts = []

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                pass

            else: 
                if mixing_range[0] <= i <= mixing_range[1]:
                    if bin_idx != [0,0]:
                        pixin = [pix,pix2]
                        style=style2
                    else:
                        style=style2
                        pixin = pix2
                                                                       
                else:
                    style = style
                    pixin= pix
                    
                    bin_idx = [0,0]
            if i > 0 and step > 0:
                out_prev = out
            
            if i<7:
                if return_att:
                    if loc_idx ==i:
                        out,a1,a2 = conv(out,styles[2*i],styles[2*i+1],pixs[2*i],pixs[2*i+1],bin_idx=bin_idx,loc=loc,return_att=return_att,add_val=add_val,loc_id = loc_id,sub=sub)
                    else:
                        out,a1,a2= conv(out,styles[2*i],styles[2*i+1],pixs[2*i],pixs[2*i+1],bin_idx=bin_idx,loc=None,return_att=return_att)
                    atts.append(a1)
                    atts.append(a2)
                else:
                    out = conv(out,styles[2*i],styles[2*i+1],pixs[2*i],pixs[2*i+1],bin_idx=bin_idx,loc=None,return_att=return_att)
            else:
                out = conv(out,styles[2*i],styles[2*i+1],pixs[2*i],pixs[2*i+1],bin_idx=bin_idx,loc=None,return_att=return_att)

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break
        if return_att:
            return out,atts
        else:
            return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)
        layers2 = [PixelNorm()]
        for i in range(n_mlp):
            layers2.append(EqualLinear(code_dim, code_dim))
            layers2.append(nn.LeakyReLU(0.2))
        self.pix = nn.Sequential(*layers2)

    def forward(
        self,
        input,
        input2,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        mean_pix=None,
        fix_style = False,
        fix_pix = False,
        style_weight=0,
        pix_weight=0,
        mixing_range=(-1, -1),
        pix_onoff= None,
        bin_idx=[0,0],loc=None,
        rand_code=False,
        intp_p=None,
        intp_s = None
    ):

        if mixing_range==(-1,-1):

            styles = self.style(input)
            pix = self.pix(input2)
            if intp_p !=None:
                pix = intp_p
            if intp_s !=None:
                styles = intp_s
            styles2 = None
            pix2 =None
            batch = input.shape[0]
        else:
            styles = self.style(input[0])
            pix = self.pix(input2[0])
            styles2 = self.style(input[1])
            pix2 = self.pix(input2[1])
            batch = input[0].shape[0]
            
            if intp_p !=None:
                    pix2 = intp_p
#                     pix2 = self.pix(input2[1])
            if intp_s !=None:
                styles2 = intp_s
            
            if rand_code:
                device = styles.device
                pix2 += torch.randn(batch,pix.shape[1]).to(device)
        
            
            
        
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
        
        return self.generator(styles, pix, step, alpha, styles2,pix2,mixing_range=mixing_range,use_ppl=use_ppl,bin_idx=bin_idx,loc=loc)

    def mean_style(self, input,input2):
        style = self.style(input).mean(0, keepdim=True)
        pix = self.pix(input2).mean(0,keepdim=True)
        return style,pix


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
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
                make_from_rgb(16),
                make_from_rgb(32),
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

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
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

        return out
