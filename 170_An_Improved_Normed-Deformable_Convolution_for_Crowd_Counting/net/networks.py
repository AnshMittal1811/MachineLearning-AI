from net.CSRNet_deform_var import CSRNet_deform_var
import torch
import torch.nn as nn
from torch.nn import init
import functools
#from net.CSRNet_deform import CSRNet_deform
from net.CSRNet_deform_var import CSRNet_deform_var
#from net.CSRNet_aspp import CSRNet_aspp
from net.ResNet import Res50
from net.HRNet.hrnet_w40_crop import HighResolutionNet as HRNet_40_crop
from net.HRNet.hrnet_w40_crop_relu import HighResolutionNet as HRNet_40_crop_relu

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def select_optim(net, opt):
    optimizer = torch.optim.Adam(net.parameters(), opt.lr, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
    return optimizer

def init_net(net, init_type='normal', init_gain=0.01, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # Has been initlized inside
    return net

def define_net(opt):
    net_name = opt.net_name
    if net_name == 'csrnet_deform':
        net = CSRNet_deform(opt.extra_loss, opt.n_deform_layer)
    elif net_name == 'csrnet_deform_var':
        net = CSRNet_deform_var(extra_loss=opt.extra_loss, n_deform_layer=opt.n_deform_layer)
    elif net_name == 'csrnet_aspp':
        net = CSRNet_aspp()
    elif net_name == 'ResNet':
        net = Res50(extra_loss=opt.extra_loss)
    elif net_name == 'hrnet_leaky': # only for  SHB dataset
        net = HRNet_40_crop(leaky_relu=True, leaky_scale=opt.leaky_scale)
        net.init_weights('hrnetv2_w40_imagenet_pretrained.pth')
    elif net_name == 'hrnet_relu':
        net = HRNet_40_crop_relu()
        net.init_weights('hrnetv2_w40_imagenet_pretrained.pth')
    else:
        raise NotImplementedError('Unrecognized model: '+net_name)
    return net
