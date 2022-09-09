import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
import math
import scipy
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
import vgg_net
# from sampler import InfiniteSamplerWrapper
from math import log, sqrt, pi
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
logabs = lambda x: torch.log(torch.abs(x))

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero. reshape
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    # feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat





# feature-level AdaIN
class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, content, style):
        assert (content.size()[:2] == style.size()[:2])
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    


def rgb2yuv(rgb):
    # B,_,_,_ = rgb.size()
    rgb_ = rgb.transpose(1,3)                              # input is 3*n*n   default
    A = torch.tensor([[0.299, -0.14714119,0.61497538], 
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]]).cuda()   # from  Wikipedia
    yuv = torch.tensordot(rgb_,A,1).transpose(1,3)
    return yuv

def yuv2rgb(yuv):
    yuv_ = yuv.transpose(1,3)                              # input is 3*n*n   default
    A = torch.tensor([[1., 1.,1.], 
                      [0., -0.39465, 2.03211],
                      [1.13983, -0.58060, 0]]).cuda()           # from  Wikipedia
    rgb = torch.tensordot(yuv_,A,1).transpose(1,3)
    return rgb


class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """ 
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)), 
            nn.Conv2d(channels, channels, self.kernel_size, 
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input 
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()
            
        return self.op(x)
    
    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2 # same dimension after padding
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) # remember this dimension

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class SplattingBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SplattingBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels,out_channels,3, 1)
        self.adain = AdaIN()
    def forward(self,c,s):
        c = F.relu(self.conv1(c))
        s = F.relu(self.conv1(s))
        sed = self.adain(c,s)
        return sed



def feature_moments_caculation(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    # the first order
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)

    # the second order
    feat_size = 2
    # N, C = size[:2]
    feat_p2 = torch.abs(feat-feat_mean).pow(feat_size).view(N, C, -1)
    N, C,L = feat_p2.shape
    feat_p2 = feat_p2.sum(dim=2)/L
    feat_p2 = feat_p2.pow(1/feat_size).view(N, C, 1)

    return feat_mean.view(N, C), feat_p2.view(N, C)





class YUVStyleNet(nn.Module):
    
    def __init__(self, in_planes):
        super(YUVStyleNet, self).__init__()

        vgg = vgg_net.vgg
        vgg.load_state_dict(torch.load('./vgg_normalised.pth'))
        self.encoder = vgg_net.Net(vgg)
        self.encoder.eval()

        self.blurer = GaussianBlurLayer(3, 5)
        self.adin = AdaIN()
        self.alpha = nn.Parameter(torch.ones(1, in_planes, 1, 1))
        self.SB1 = SplattingBlock(3,3)
        self.SB2 = SplattingBlock(64,64) 
        self.SB3 = SplattingBlock(128, 128)
        self.SB4 = SplattingBlock(256, 256)
        self.SB5 = SplattingBlock(512, 512)
        self.conv_up5 = nn.Sequential(
            nn.Conv2d(512, 16,kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 256,kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.conv_up4 = nn.Sequential(
            nn.Conv2d(512, 16,kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 128,kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(256, 16,kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64,kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.conv_up2 = nn.Sequential(
            nn.Conv2d(128, 16,kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3,kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.conv_up1 = nn.Sequential(
            nn.Conv2d(6, 16,kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3,kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, content, style):

        B,C,H,W = content.size()

        resize_style = torch.nn.functional.interpolate(style,(256, 256), mode='bilinear', align_corners=False)
        resize_content = torch.nn.functional.interpolate(content,(256, 256), mode='bilinear', align_corners=False)


        style_feats = self.encoder.encode_with_intermediate(resize_style)
        content_feat = self.encoder.encode_with_intermediate(resize_content)


        stylized5  = self.SB5(content_feat[-1],style_feats[-1])
        stylized5 = self.conv_up5(stylized5)
        stylized5 = torch.nn.functional.interpolate(stylized5, scale_factor=2, mode='bilinear', align_corners=False)

        stylized4  = self.SB4(content_feat[-2],style_feats[-2])
        stylized4 = torch.cat((stylized5, stylized4), 1)
        stylized4 = self.conv_up4(stylized4)
        stylized4 = torch.nn.functional.interpolate(stylized4, scale_factor=2, mode='bilinear', align_corners=False)

        stylized3  = self.SB3(content_feat[-3],style_feats[-3])
        stylized3 = torch.cat((stylized3, stylized4), 1)
        stylized3 = self.conv_up3(stylized3)
        stylized3 = torch.nn.functional.interpolate(stylized3, scale_factor=2, mode='bilinear', align_corners=False)

        stylized2  = self.SB2(content_feat[-4],style_feats[-4])
        stylized2 = torch.cat((stylized3, stylized2), 1)
        stylized2 = self.conv_up2(stylized2)

        stylized1  = self.SB1(resize_content,resize_style)
        stylized1 = torch.cat((stylized1, stylized2), 1)

        stylized1 = self.conv_up1(stylized1)


        stylized1 =  torch.nn.functional.interpolate(stylized1,(H, W), mode='bilinear', align_corners=False)
        
        stylized_b = self.blurer(stylized1)#模糊化
        stylized_b = rgb2yuv(stylized_b)#转为yuv
        stylized_rgb = rgb2yuv(content)#原图转为yuv
        # stylized1 = yuv2rgb(stylized_b)
        a = 1.0
        stylized_rgb= self.alpha*self.adin(stylized_rgb,stylized_b)
        stylized_rgb[:,1] = a*stylized_b[:,1]+(1-a)*stylized_rgb[:,1]#风格图中的uv转换到原图
        stylized_rgb[:,2] = a*stylized_b[:,2]+(1-a)*stylized_rgb[:,2]#风格图中的uv转换到原图
        output = yuv2rgb(stylized_rgb)#将转换后的图转为rgb
        return stylized1, output

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)

        bs, ch = input.size()[:2]
        input = input.view(bs, ch, -1)
        target = target.view(bs, ch, -1)
        input_mean, input_std = feature_moments_caculation(input)
        target_mean, target_std  = feature_moments_caculation(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)#+ \

    def caculate_losses(self, content_images, style_images, stylized_images):
        style_feats = self.encoder(style_images)#style_images[2, 3, 256, 256];4
        content_feat = self.encoder(content_images)#content_feat[2, 512, 32, 32]
        stylized_feats = self.encoder(stylized_images)

        loss_c = self.calc_content_loss(stylized_feats[-1], content_feat[-1])
        loss_s = self.calc_style_loss(stylized_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i])
        return loss_c, loss_s



        
if __name__ == '__main__':
    USE_CUDA = True

    from PIL import Image
    import numpy as np

    import time
    model = YUVStyleNet(3).cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    content_images=style_images=torch.randn(1,3,512,512).cuda()
    start = time.time()
    times_per_img = 500
    for i in range(times_per_img):
        stylized,output=model(content_images,style_images) #codewords 8,32,16;reconstruction 8,3,2048
        loss_c, loss_s = model.caculate_losses(content_images, style_images, stylized)
        # print(out)
    print("time:", (time.time() - start) / times_per_img)