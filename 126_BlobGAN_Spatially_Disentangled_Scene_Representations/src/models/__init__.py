import sys
from typing import Any, Tuple, Dict

from pytorch_lightning import LightningModule

from models import networks
from utils import to_dataclass_cfg
# from .segmenter import *
from .blobgan import *
from .gan import *
from .invertblobgan import *


def get_model(name: str, return_cfg: bool = False, **kwargs) -> Tuple[LightningModule, Dict[str, Any]]:
    cls = getattr(sys.modules[__name__], name)
    cfg = to_dataclass_cfg(kwargs, cls)
    if return_cfg:
        return cls(**cfg), cfg
    else:
        return cls(**cfg)
