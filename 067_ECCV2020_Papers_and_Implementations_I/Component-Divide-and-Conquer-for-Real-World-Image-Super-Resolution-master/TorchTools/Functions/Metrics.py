import numpy as np
try:
    from math import log10
except ImportError:
    from math import log
    def log10(x):
        return log(x) / log(10.)

import torch
import math
from .functional import to_tensor
from PIL import Image

def mse(x, y):
    """
    MSE Error
    :param x: tensor / numpy.ndarray
    :param y: tensor / numpy.ndarray
    :return: float
    """
    diff = x - y
    diff = diff * diff
    if isinstance(diff, np.ndarray):
        diff = torch.FloatTensor(diff)
    return torch.mean(diff)


def psnr(x, y, peak=1.):
    """
    psnr from tensor
    :param x: tensor
    :param y: tensor
    :return: float (mse, psnr)
    """
    _mse = mse(x, y)
    # return _mse, 10 * log10((peak ** 2) / _mse)
    return 10 * log10((peak ** 2) / _mse)

def PSNR(x, y, c=-1):
    """
    PSNR from PIL.Image / tensor
    :param x: PIL.Image
    :param y: PIL.Image
    :return: float (mse, psnr)
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return psnr(x, y, peak=1.)
    else:
        if c == -1:
            return psnr(to_tensor(x), to_tensor(y), peak=1.)
        else:
            return psnr(to_tensor(x)[c], to_tensor(y)[c], peak=1.)


def YCbCr_psnr(sr, hr, scale, peak=1.):
    """
    Caculate PSNR in YCbCr`s Y Channel
    :param sr:
    :param hr:
    :param scale:
    :return:
    """
    diff = (sr - hr) / peak
    shave = scale
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

