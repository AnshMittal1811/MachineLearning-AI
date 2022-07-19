import logging
import math
import torch
import torch.nn as nn

BN = None

__all__ = ['resnet50','resnetal50','resnet101']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
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
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, with_neck=False,  pretrain_path=None, enable_fc=False):
        super(ResNet, self).__init__()

        global BN

        def BNFunc(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs)
        BN = BNFunc

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_neck:
            self.neck = block(512 * block.expansion, 512)
        else:
            self.neck = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.enable_fc = enable_fc
        # if enable_fc:
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

        if pretrain_path is not None:
            load_state_vision_model(self, pretrain_path)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        if self.neck is not None:
            layer4 = self.neck(layer4)
        
        x = self.avgpool(layer4)
        x = torch.flatten(x, 1)
        # if self.enable_fc:
        #     x = self.fc(x)

        return x




def load_state_vision_model(model, ckpt_path):
    ckpt_state = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt_state:
        # our upstream converted checkpoint
        ckpt_state = ckpt_state['state_dict']
        prefix = ''
    elif 'model' in ckpt_state:
        # prototype checkpoint
        ckpt_state = ckpt_state['model']
        prefix = 'module.'
    else:
        # official checkpoint
        prefix = ''

    logger = logging.getLogger('global')
    if ckpt_state:
        logger.info('==> Loading model state "{}XXX" from pre-trained model..'.format(prefix))
        
        own_state = model.state_dict()
        state = {}
        for name, param in ckpt_state.items():
            if name.startswith(prefix):
                state[name[len(prefix):]] = param
        success_cnt = 0
        for name, param in state.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    if isinstance(param, bool):
                        own_state[name] = param
                    else:
                        # normal version 
                        own_state[name].copy_(param)
                    success_cnt += 1
                except Exception as err:
                    logger.warn(err)
                    logger.warn('while copying the parameter named {}, '
                                         'whose dimensions in the model are {} and '
                                         'whose dimensions in the checkpoint are {}.'
                                         .format(name, own_state[name].size(), param.size()))
                    logger.warn("But don't worry about it. Continue pretraining.")
        ckpt_keys = set(state.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        logger.info('Successfully loaded {} key(s) from {}'.format(success_cnt, ckpt_path))
        for k in missing_keys:
            logger.warn('Caution: missing key from checkpoint: {}'.format(k))
        redundancy_keys = ckpt_keys - own_keys
        for k in redundancy_keys:
            logger.warn('Caution: redundant key from checkpoint: {}'.format(k))



def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3],**kwargs)
    return model


def resnetal50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNetAL(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return model


