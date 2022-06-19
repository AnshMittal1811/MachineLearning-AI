import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision.ops as ops
import time
import torch



class CSRNet_deform_var(nn.Module):
    def __init__(self, extra_loss=0, n_deform_layer=6, deform_dilation=2):
        super(CSRNet_deform_var, self).__init__()
        self.dd = deform_dilation
        self.extra_loss = extra_loss
        #self.scale_loss = scale_loss
        self.n_dilated = 6 - n_deform_layer
        self.backend_feat_1 = (512, 2)
        self.backend_feat_2 = (512, 2)
        self.backend_feat_3 = (512, 2)
        self.backend_feat_4 = (512, 2)
        self.backend_feat_5 = (128, 2)
        self.backend_feat_6 = (64, 2)
        self.front_end = nn.Sequential(*(list(list(models.vgg16_bn(True).children())[0].children())[0:33]))

        # normal dilated convs
        for j in range(1, self.n_dilated + 1):
            if j == 1:
                in_c = 512
                out_c = self.backend_feat_1[0]
            else:
                in_c = getattr(self, 'backend_feat_{:d}'.format(j-1))[0]
                out_c = getattr(self, 'backend_feat_{:d}'.format(j))[0]

            which_backend_feat = getattr(self, 'backend_feat_{:d}'.format(j))
            setattr(self, 'dconv_{:d}'.format(j), make_layers(which_backend_feat, in_channels=in_c, batch_norm=True))

        for i in range(self.n_dilated + 1, 7):
            if i == 1:
                in_c = 512
                out_c = self.backend_feat_1[0]
            else:
                in_c = getattr(self, 'backend_feat_{:d}'.format(i-1))[0]
                out_c = getattr(self, 'backend_feat_{:d}'.format(i))[0]

            setattr(self, 'offset_w_{:d}'.format(i), nn.Conv2d(in_channels=in_c, out_channels=2*3*3, kernel_size=3, padding=1))
            #setattr(self, 'scale_w_{:d}'.format(i), nn.Conv2d(in_channels=in_c, out_channels=2*3*3, kernel_size=3, padding=1))

            # In pytorch1.8, the weight of deformable has been included into ops.DeformConv2d (So, it is good..)
            setattr(self, 'deform_{:d}'.format(i), ops.DeformConv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=self.dd, dilation=self.dd))
            setattr(self, 'bn_{:d}'.format(i), nn.BatchNorm2d(out_c))

        # add bn after
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        for m in self.output_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, out_feat=False):
        img_shape = x.shape
        x = self.front_end(x)
        # forward dilated convs
        for i in range(1, self.n_dilated + 1):
            cur_block = getattr(self, 'dconv_{:d}'.format(i))
            x = cur_block(x)

        x_offset_list = []
        # add loss contrain on the offset
        for j in range(self.n_dilated + 1, 7):
            cur_offset = getattr(self, 'offset_w_{:d}'.format(j))
            cur_deform = getattr(self, 'deform_{:d}'.format(j))
            cur_bn = getattr(self, 'bn_{:d}'.format(j))
            x_offset = cur_offset(x)
            x_offset = torch.tanh(x_offset)
            x = F.relu_(cur_bn(cur_deform(x, x_offset)))
        output = self.output_layer(x) # add relu for SHB for now, later, SHA also relu
        # no this line of code in the original csrnet
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        if (self.extra_loss) and out_feat == True:
            return output, x_offset_list
        else:
            return output


def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    cfg = [cfg]
    for v, atrous in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=atrous, dilation=atrous)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
