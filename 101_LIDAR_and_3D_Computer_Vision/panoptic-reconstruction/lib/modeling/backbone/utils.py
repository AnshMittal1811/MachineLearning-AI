from torch import nn

from lib.config import config
from lib.modeling import backbone


def build_backbone() -> nn.Module:
    num_layers = config.MODEL.BACKBONE.CONV_BODY.split('-')[1]
    resnet_network = {
        "18": backbone.resnet.resnet18,
        "34": backbone.resnet.resnet34,
        "50": backbone.resnet.resnet50,
        "101": backbone.resnet.resnet101
    }

    model = resnet_network[num_layers](config.MODEL.PRETRAIN)

    return model
