import importlib
import random
from dataclasses import is_dataclass, fields
from math import pi
from typing import Any, Union, TypeVar, Tuple, Optional, Dict, OrderedDict

import einops
import numpy as np
import opt_einsum as oe
import torch
from PIL import Image, ImageDraw
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

from utils.io import load_pretrained_weights
from .distributed import is_rank_zero

T = TypeVar('T')
FromConfig = Union[T, Dict[str, Any]]
NTuple = Tuple[T, ...]
StateDict = OrderedDict[str, torch.Tensor]

TORCH_EINSUM = True
einsum = torch.einsum if TORCH_EINSUM else oe.contract


def recursive_compare(d1: dict, d2: dict, level: str = 'root') -> str:
    ret = []
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            ret.append('{:<20} - {} + {}'.format(level, ','.join(s1 - s2), ','.join(s2 - s1)))
            common_keys = s1 & s2
        else:
            common_keys = set(d1.keys())

        for k in common_keys:
            ret.append(recursive_compare(d1[k], d2[k], level='{}.{}'.format(level, k)))
    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            ret.append('{:<20} len1={}; len2={}'.format(level, len(d1), len(d2)))
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            ret.append(recursive_compare(d1[i], d2[i], level='{}[{}]'.format(level, i)))
    else:
        if d1 != d2:
            ret.append('{:<20} {} -> {}'.format(level, d1, d2))
    return '\n'.join(filter(None, ret))


def import_external(name: str, pretrained: Optional[Union[str, DictConfig]] = None, **kwargs):
    module, name = name.rsplit('.', 1)
    ret = getattr(importlib.import_module(module), name)
    ret = ret(**to_dataclass_cfg(kwargs, ret))
    return load_pretrained_weights(name, pretrained, ret)


def run_at_step(step: int, freq: int):
    return (freq > 0) and ((step + 1) % freq == 0)


def rotation_matrix(theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack([cos, sin, -sin, cos], dim=-1).view(*theta.shape, 2, 2)


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)

    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def splat_features_from_scores(scores: Tensor, features: Tensor, size: Optional[int],
                               channels_last: bool = True) -> Tensor:
    """

    Args:
        channels_last: expect input with M at end or not, see below
        scores: [N, H, W, M] (or [N, M, H, W] if not channels last)
        features: [N, M, C]
        size: dimension of map to return
    Returns: [N, C, H, W]

    """
    if size and not (scores.shape[2] == size):
        if channels_last:
            scores = einops.rearrange(scores, 'n h w m -> n m h w')
        scores = F.interpolate(scores, size, mode='bilinear', align_corners=False)
        einstr = 'nmhw,nmc->nchw'
    else:
        einstr = 'nhwm,nmc->nchw' if channels_last else 'nmhw,nmc->nchw'
    return einsum(einstr, scores, features).contiguous()


def cholesky_to_matrix(covs: Tensor) -> Tensor:
    covs[..., ::3] = covs[:, :, ::3].exp()
    covs[..., 2] = 0
    covs = einops.rearrange(covs, 'n m (x y) -> n m x y', x=2, y=2)
    covs = einsum('nmji,nmjk->nmik', covs, covs)  # [n, m, 2, 2]
    return covs


def jitter_image_batch(images: Tensor, dy: int, dx: int) -> Tensor:
    # images: N x C x H x W
    images = torch.roll(images, (dy, dx), (2, 3))
    if dy > 0:
        images[:, :, :dy, :] = 0
    else:
        images[:, :, dy:, :] = 0
    if dx > 0:
        images[:, :, :, :dx] = 0
    else:
        images[:, :, :, dx:] = 0
    return images


DERANGEMENT_WARNED = False


def derangement(n: int) -> Tensor:
    global DERANGEMENT_WARNED
    orig = torch.arange(n)
    shuffle = torch.randperm(n)
    if n == 1 and not DERANGEMENT_WARNED:
        if is_rank_zero():
            print('Warning: called derangement with n=1!')
        DERANGEMENT_WARNED = True
    while (n > 1) and (shuffle == orig).any():
        shuffle = torch.randperm(n)
    return shuffle


def pyramid_resize(img, cutoff):
    """

    Args:
        img: [N x C x H x W]
        cutoff: threshold at which to stop pyramid

    Returns: gaussian pyramid

    """
    out = [img]
    while img.shape[-1] > cutoff:
        img = F.interpolate(img, img.shape[-1] // 2, mode='bilinear', align_corners=False)
        out.append(img)
    return {i.size(-1): i for i in out}


def derange_tensor(x: Tensor, dim: int = 0) -> Tensor:
    if dim == 0:
        return x[derangement(len(x))]
    elif dim == 1:
        return x[:, derangement(len(x[0]))]


def derange_tensor_n_times(x: Tensor, n: int, dim: int = 0, stack_dim: int = 0) -> Tensor:
    return torch.stack([derange_tensor(x, dim) for _ in range(n)], stack_dim)


def to_dataclass_cfg(cfg, cls):
    """
    Can't add **kwargs catch-all to dataclass, so need to strip dict of keys that are not fields
    """
    if is_dataclass(cls):
        return {k: v for k, v in cfg.items() if k in [f.name for f in fields(cls)]}
    return cfg


def random_polygons(size: int, shape: Union[int, Tuple[int, ...]]):
    if type(shape) is int:
        shape = (shape,)
    n = np.prod(shape)
    return torch.stack([random_polygon(size) for _ in range(n)]).view(*shape, 1, size, size)


def random_polygon(size: int):
    # Logic from Copy Paste GAN
    img = Image.new("RGB", (size, size), "black")
    f = lambda s: round(size * s)
    to_xy = lambda r, θ, p: (p + r * np.array([np.cos(θ), np.sin(θ)]))
    c = np.array([random.randint(f(0.1), f(0.9)), random.randint(f(0.1), f(0.9))])
    n_vert = random.randint(4, 6)
    coords = []
    while len(coords) < n_vert:
        coord = to_xy(random.uniform(f(0.1), f(0.5)), random.uniform(0, 2 * pi), c)
        if coord.min() >= 0:
            coords.append(tuple(coord))
    ImageDraw.Draw(img).polygon(coords, fill="white")
    return to_tensor(img)[:1]
