from pathlib import Path

import PIL

from yacs.config import CfgNode as Node
import torch
from torch.utils.collect_env import get_pretty_env_info


def save_config(config: Node, save_path: Path) -> None:
    with open(save_path, "w") as file:
        file.write(config.dump())


def get_pil_version() -> str:
    return "\n        Pillow ({})".format(PIL.__version__)


def collect_env_info() -> str:
    env_str = get_pretty_env_info()
    env_str += get_pil_version()
    return env_str


def re_seed(seed: int = 0) -> None:
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    from torch.backends import cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False  # not too much slower

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def count_parameters(model: torch.nn.Module, check_grad: bool = False) -> int:
    if check_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
