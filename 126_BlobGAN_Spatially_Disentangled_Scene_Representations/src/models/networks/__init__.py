import sys
from typing import Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn

from utils import import_external, is_rank_zero, get_checkpoint_path, load_pretrained_weights
from .stylegan import *
from .layoutnet import *


def get_network(name: str, pretrained: Optional[Union[str, DictConfig]] = None, **kwargs) -> nn.Module:
    if '.' in name:
        ret = import_external(name, pretrained, **kwargs)
        return ret
    else:
        ret = getattr(sys.modules[__name__], name)(**kwargs)
        return load_pretrained_weights(name, pretrained, ret)
