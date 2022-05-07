import sys

from utils import to_dataclass_cfg
from .nodata import *
from .imagefolder import *
from .multiimagefolder import *

def get_datamodule(name: str, **kwargs) -> LightningDataModule:
    cls = getattr(sys.modules[__name__], name)
    return cls(**to_dataclass_cfg(kwargs, cls))
