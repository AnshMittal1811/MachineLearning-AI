from collections import OrderedDict
import logging

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['modified_res50', 'modified_res50x16', 'modified_res101']

BN = None


def get_bn(config):
    assert not config.use_sync_bn
    def BNFunc(*args, **kwargs):
        return torch.nn.BatchNorm2d(*args, **kwargs, **config.get('kwargs', {}))
    return BNFunc


def load_clip_state_vision_model(model, ckpt_path):
    ckpt_state = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt_state:
        # our gvm-clip checkpoint
        ckpt_state = ckpt_state['state_dict']
        prefix = 'module.vision_model.'
    elif 'model' in ckpt_state:
        # prototype checkpoint
        ckpt_state = ckpt_state['model']
        prefix = 'module.visual.'
    else:
        # OpenAI checkpoint
        prefix = 'visual.'

    logger = logging.getLogger('global')
    if ckpt_state:
        logger.info('==> Loading vision model state "{}XXX" from CLIP model..'.format(prefix))
        
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
            logger.warn('Caution: missing key from vision model of CLIP checkpoint: {}'.format(k))
        redundancy_keys = ckpt_keys - own_keys
        for k in redundancy_keys:
            logger.warn('Caution: redundant key from vision model of CLIP checkpoint: {}'.format(k))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = BN(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = BN(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = BN(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", BN(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64,
                 num_classes=1000, bn=None, pretrain_path=None, enable_attnpool=False, enable_fc=False):

        global BN

        BN = get_bn(bn)

        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BN(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = BN(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = BN(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.enable_attnpool = enable_attnpool
        self.enable_fc = enable_fc
        if enable_attnpool:
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)
            
            if enable_fc:
                self.fc = nn.Linear(output_dim, num_classes)
        else:
            self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            if enable_fc:
                self.fc = nn.Linear(embed_dim, num_classes)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

        if pretrain_path is not None:
            load_clip_state_vision_model(self, pretrain_path)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.enable_attnpool:
            x = self.attnpool(x)
        else:
            x = self.global_avgpool(x).squeeze(2).squeeze(2)
        if self.enable_fc:
            x = self.fc(x)

        return x


def modified_res50(**kwargs):
    default_kwargs = {
        'layers': [3, 4, 6, 3],
        'output_dim': 1024,   # keep same with text transformer
        'heads': 32,
        'input_resolution': 224,
        'width': 64,
    }
    default_kwargs.update(**kwargs)
    return ModifiedResNet(**default_kwargs)

def modified_res101(**kwargs):
    default_kwargs = {
        'layers': [3, 4, 23, 3],
        'output_dim': 512,   # keep same with text transformer
        'heads': 32,
        'input_resolution': 224,
        'width': 64,
    }
    default_kwargs.update(**kwargs)
    return ModifiedResNet(**default_kwargs)

def modified_res50x16(**kwargs):
    default_kwargs = {
        'layers': [6, 8, 18, 8],
        'output_dim': 768,   # keep same with text transformer
        'heads': 48,
        'input_resolution': 384,
        'width': 96,
    }
    default_kwargs.update(**kwargs)
    return ModifiedResNet(**default_kwargs)

# heads = vision_width // 64
# input_resolution = image_resolution,
# patch_size = vision_patch_size,
# width = vision_width,
# layers = vision_layers,
# heads = vision_heads,
# output_dim = embed_dim

# ('CLIP-RN50', lambda *args, **kwargs: CLIP(1024, 224, (3, 4, 6, 3), 64)),
# ('CLIP-RN50x16', lambda *args, **kwargs: CLIP(768, 384, (6, 8, 18, 8), 96)),
# ('CLIP-RN101', lambda *args, **kwargs: CLIP(512, 224, (3, 4, 23, 3), 64)),