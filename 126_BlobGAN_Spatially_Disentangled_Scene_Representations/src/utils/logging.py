from typing import Any, Union, Optional, Tuple, Dict, List
import torch
from numbers import Number

from omegaconf import DictConfig
from torch import Tensor
from .distributed import is_rank_zero


def scalars_to_log_dict(scalars: Dict[Any, Union[Number, Tensor]], mode: str) -> Dict[str, Number]:
    return {f'{mode}/{k}': (v.item() if isinstance(v, Tensor) else v) for k, v in scalars.items()}


def epoch_outputs_to_log_dict(outputs: List[Dict[str, Tensor]],
                              n_max: Optional[Union[int, str]] = None,
                              shuffle: bool = False,
                              reduce: Optional[str] = None) -> Dict[str, Tensor]:
    # Converts list of dicts (per-batch return values) into dict of concatenated list element dict values
    # Optionally return a tensor of length at most n_max for each key, and shuffle
    # If n_max is "batch", return one batch worth of tensors
    # Either cat or stack, depending on whether batch output is 0-d tensor (scalar) or not
    def merge_fn(v):
        return (torch.cat if len(v.shape) else torch.stack) if torch.is_tensor(v) else Tensor

    reduce_fn = lambda x: x
    if reduce is not None:
        if reduce == 'mean':
            reduce_fn = torch.mean
        elif reduce == 'sum':
            reduce_fn = torch.sum
        else:
            raise ValueError('reduce must be either `mean` or `sum`')
    out_dict = {k: reduce_fn(merge_fn(v)([o[k] for o in outputs])) for k, v in outputs[0].items() if v is not None}
    if n_max is not None:
        for k, v in out_dict.items():
            if shuffle:
                v = v[torch.randperm(len(v))]
            n_max_ = len(outputs[0][k]) if n_max == "batch" else n_max
            out_dict[k] = v[:n_max_]
    return out_dict


def scale_logging_rates(d: DictConfig, c: Number, strs: Tuple[str] = ('log', 'every_n_steps'), prefix: str = 'config'):
    if c == 1:
        return
    for k, v in d.items():
        if all([s in k for s in strs]):
            d[k] = type(v)(v * c)
            if is_rank_zero():
                print(f'Scaling {prefix}.{k} from {v} to {type(v)(v * c)} due to gradient accumulation')
        elif isinstance(v, DictConfig):
            scale_logging_rates(v, c, strs, prefix=prefix + '.' + k)
