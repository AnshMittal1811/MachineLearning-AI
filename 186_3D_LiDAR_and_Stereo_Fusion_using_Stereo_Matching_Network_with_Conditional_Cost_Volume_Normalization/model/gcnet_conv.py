"""
Ref: https://github.com/wyf2017/DSMnet/blob/master/models/util_conv.py
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from . import ccvnorm

flag_check_shape = False
flag_bn = False
flag_bias_default = True
activefun_default = nn.ReLU(inplace=True)


def msg_conv(obj_in, obj_conv, obj_out):
    return "\n input: %s\n conv: %s\n output: %s\n" % (str(obj_in.shape), str(obj_conv), str(obj_out.shape))


def msg_shapes(**args):
    n = len(args)
    msg = ""
    shapes = [str(arg.shape) for arg in args]
    for i in range(n-1):
        msg += "%s--->%s\n" % (shapes[i], shapes[i+1])
    return msg


# weight init
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = fanin_init(m.weight.data.size())
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# corr1d
class Corr1d(nn.Module):
    def __init__(self, kernel_size=1, stride=1, D=1, simfun=None):
        super(Corr1d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.D = D
        if(simfun is None):
            self.simfun = self.simfun_default
        else: # such as simfun = nn.CosineSimilarity(dim=1)
            self.simfun = simfun
    
    def simfun_default(self, fL, fR):
        return (fL*fR).sum(dim=1)
        
    def forward(self, fL, fR):
        bn, c, h, w = fL.shape
        D = self.D
        stride = self.stride
        kernel_size = self.kernel_size
        corrmap = Variable(torch.zeros(bn, D, h, w).type_as(fL.data))
        corrmap[:, 0] = self.simfun(fL, fR)
        for i in range(1, D):
            if(i >= w): break
            idx = i*stride
            corrmap[:, i, :, idx:] = self.simfun(fL[:, :, :, idx:], fR[:, :, :, :-idx])
        if(kernel_size>1):
            assert kernel_size%2 == 1
            m = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
            corrmap = m(corrmap)
        return corrmap


class Conv2d(nn.Conv2d):
    def forward(self, obj_in):
        obj_out = super(Conv2d, self).forward(obj_in)
        if(flag_check_shape):
            print(msg_conv(obj_in, self, obj_out))
        return obj_out


class ConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, obj_in):
        obj_out = super(ConvTranspose2d, self).forward(obj_in)
        if(flag_check_shape):
            print(msg_conv(obj_in, self, obj_out))
        return obj_out


class Conv3d(nn.Conv3d):
    def forward(self, obj_in):
        obj_out =super(Conv3d, self).forward(obj_in)
        if(flag_check_shape):
            print(msg_conv(obj_in, self, obj_out))
        return obj_out


class ConvTranspose3d(nn.ConvTranspose3d):
    def forward(self, obj_in):
        obj_out = super(ConvTranspose3d, self).forward(obj_in)
        if(flag_check_shape):
            print(msg_conv(obj_in, self.conv, obj_out))
        return obj_out


class conv3d_ccvnorm(nn.Module):
    """ 3d deconvolution with padding, bn and activefun """
    def __init__(self, in_planes, out_planes, D, kernel_size=3, stride=1, flag_bias=flag_bias_default, bn=flag_bn, 
                 activefun=activefun_default, mode='categorical', norm_in_channels=1):
        super(conv3d_ccvnorm, self).__init__()
        self.conv3d = Conv3d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1)//2, bias=flag_bias)
        self.mode = mode
        if self.mode == 'naive_categorical':
            bn_fn = lambda: ccvnorm.NaiveCategoricalConditionalBatchNorm(out_planes, D)
        elif self.mode == 'naive_continuous':
            bn_fn = lambda: ccvnorm.NaiveContinuousConditionalBatchNorm(out_planes, D)
        elif self.mode == 'categorical':
            bn_fn = lambda: ccvnorm.CategoricalConditionalCostVolumeNorm(out_planes, D)
        elif self.mode == 'continuous':
            bn_fn = lambda: ccvnorm.ContinuousConditionalCostVolumeNorm(out_planes, D)
        elif self.mode == 'categorical_hier':
            bn_fn = lambda: ccvnorm.CategoricalHierConditionalCostVolumeNorm(out_planes, D)
        else:
            raise NotImplementedError
        if bn: self.bn_layer = bn_fn()
        if activefun: self.active_fn = activefun
        self.bn = bn
        self.activefun = activefun
    
    def forward(self, x, c, feats=None):
        x = self.conv3d(x)
        if self.bn:
            x = self.bn_layer(x, c)
        if self.activefun: x = self.active_fn(x)
        return x


class deconv3d_ccvnorm(nn.Module):
    """ 3d deconvolution with padding, bn and activefun """
    def __init__(self, in_planes, out_planes, D, kernel_size=4, stride=2, flag_bias=flag_bias_default, bn=flag_bn, 
                 activefun=activefun_default, mode='categorical', norm_in_channels=1):
        super(deconv3d_ccvnorm, self).__init__()
        assert stride > 1
        p = (kernel_size - 1)//2
        op = stride - (kernel_size - 2*p)
        self.conv3d = ConvTranspose3d(in_planes, out_planes, kernel_size, stride, padding=p, output_padding=op, bias=flag_bias)
        self.mode = mode
        if self.mode == 'naive_categorical':
            bn_fn = lambda: ccvnorm.NaiveCategoricalConditionalBatchNorm(out_planes, D)
        elif self.mode == 'naive_continuous':
            bn_fn = lambda: ccvnorm.NaiveContinuousConditionalBatchNorm(out_planes, D)
        elif self.mode == 'categorical':
            bn_fn = lambda: ccvnorm.CategoricalConditionalCostVolumeNorm(out_planes, D)
        elif self.mode == 'continuous':
            bn_fn = lambda: ccvnorm.ContinuousConditionalCostVolumeNorm(out_planes, D)
        elif self.mode == 'categorical_hier':
            bn_fn = lambda: ccvnorm.CategoricalHierConditionalCostVolumeNorm(out_planes, D)
        else:
            raise NotImplementedError
        if bn: self.bn_layer = bn_fn()
        if activefun: self.active_fn = activefun
        self.bn = bn
        self.activefun = activefun
    
    def forward(self, x, c, feats=None):
        x = self.conv3d(x)
        if self.bn:
            x = self.bn_layer(x, c)
        if self.activefun: x = self.active_fn(x)
        return x

        
def conv2d_bn(in_planes, out_planes, kernel_size=3, stride=1, flag_bias=flag_bias_default, bn=flag_bn, activefun=activefun_default):
    """ 2d convolution with padding, bn and activefun """
    assert kernel_size % 2 == 1
    conv2d = Conv2d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1)//2, bias=flag_bias)
    
    if(not bn and not activefun): 
        return conv2d
    
    layers = []
    layers.append(conv2d)
    if bn: layers.append(nn.BatchNorm2d(out_planes))
    if activefun: layers.append(activefun)
    
    return nn.Sequential(*layers)


def deconv2d_bn(in_planes, out_planes, kernel_size=4, stride=2, flag_bias=flag_bias_default, bn=flag_bn, activefun=activefun_default):
    """ 2d deconvolution with padding, bn and activefun """
    assert stride > 1
    p = (kernel_size - 1)//2
    op = stride - (kernel_size - 2*p)
    conv2d = ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding=p, output_padding=op, bias=flag_bias)
    
    if(not bn and not activefun): 
        return conv2d
    
    layers = []
    layers.append(conv2d)
    if bn: layers.append(nn.BatchNorm2d(out_planes))
    if activefun: layers.append(activefun)
    
    return nn.Sequential(*layers)


def conv3d_bn(in_planes, out_planes, kernel_size=3, stride=1, flag_bias=flag_bias_default, bn=flag_bn, activefun=activefun_default):
    """ 3d convolution with padding, bn and activefun """
    conv3d = Conv3d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1)//2, bias=flag_bias)

    if(not bn and not activefun): 
        return conv3d

    layers = []
    layers.append(conv3d)
    if bn: layers.append(nn.BatchNorm3d(out_planes))
    if activefun: layers.append(activefun)

    return nn.Sequential(*layers)


def deconv3d_bn(in_planes, out_planes, kernel_size=4, stride=2, flag_bias=flag_bias_default, bn=flag_bn, activefun=activefun_default):
    """ 3d deconvolution with padding, bn and activefun """
    assert stride > 1
    p = (kernel_size - 1)//2
    op = stride - (kernel_size - 2*p)
    conv2d = ConvTranspose3d(in_planes, out_planes, kernel_size, stride, padding=p, output_padding=op, bias=flag_bias)
    
    if(not bn and not activefun): 
        return conv2d
    
    layers = []
    layers.append(conv2d)
    if bn: layers.append(nn.BatchNorm3d(out_planes))
    if activefun: layers.append(activefun)
    
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def make_layer_res(block, blocks, inplanes, planes, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


def conv_res(inplanes, planes, blocks, stride=1):
    block = BasicBlock
    return make_layer_res(block, blocks, inplanes, planes, stride)


def conv_res_bottleneck(inplanes, planes, blocks, stride=1):
    block = Bottleneck
    return make_layer_res(block, blocks, inplanes, planes, stride)


def test():
    net = conv_res(3, 32, blocks=2, stride=1).cuda()
    print(net)
    tmp = net[0].conv1.weight.data*1
    net_init(net)
    print((tmp - net[0].conv1.weight.data).max())
    im = Variable(torch.randn(1,3,256, 128)).cuda()
    y= net(im)
    print(y.shape)
    
    net = conv_res_bottleneck(3, 32, blocks=2, stride=1).cuda()
    print(net)
    net_init(net)
    im = Variable(torch.randn(1,3,256, 128)).cuda()
    y= net(im)
    print(y.shape)
