# coding: utf-8
import numpy as np
import random
import functools

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io

try:
    from math import log2
except:
    from math import log
    def log2(x):
        return log(x) / log(2)

import math

from .modules import residualBlock, upsampleBlock, DownsamplingShuffle, Attention, Flatten

isotropic_kernel_label = 0
anisotropic_kernel_label = 1

def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.25, sig_max=1.75, scaling=3, l=15, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return (k, x, y, pi)


def random_isotropic_gaussian_kernel(sig_min=0.25, sig_max=1.75, l=15, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return (k, x, x, 0)


def random_gaussian_kernel(l=15, sig_min=0.25, sig_max=1.75, rate_iso=0.3, scaling=3, tensor=False):
    """
    return gaussian kernel and corresponding params, -1 for isotropic, 1 for anisotropic
    :param l:
    :return:
    """
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def random_batch_kernel_param(batch, l=15, sig_min=0.25, sig_max=1.75, rate_iso=0.3, scaling=3, tensor=True):
    """
    Generating random_batch_kernel, with its corresponding params
    :param batch:
    :param l:
    :param sig_min:
    :param sig_max:
    :param rate_iso:
    :param scaling:
    :param tensor:
    :return: kernel, param[sig_x, sig_y, pi]
    """
    batch_kernel = np.zeros((batch, l, l))
    batch_kernel_param = np.zeros((batch, 3))
    for i in range(batch):
        kernel, sig_x, sig_y, pi = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
        batch_kernel[i] = kernel
        batch_kernel_param[i] = np.array([sig_x, sig_y, pi])
    return (torch.FloatTensor(batch_kernel), torch.FloatTensor(batch_kernel_param)) if tensor else (batch_kernel, batch_kernel_param)


def batch_kernel_param_aug(batch_kernel_param, sig_min=0.25, sig_max=1.75, sig_aug=0.2, scaling=3):
    """
    adding augmentation to generate larger kernel as label
    :param batch_kernel_param:
    :param sig_min:
    :param sig_max:
    :param sig_aug:
    :param scaling:
    :return: added param
    """
    augment = np.array([sig_aug, sig_aug * scaling])
    augment = torch.FloatTensor(augment).expand(batch_kernel_param.shape[0], 2)
    augmented = torch.clamp(batch_kernel_param[:, 1:3] + augment, min=sig_min, max=sig_max)
    batch_kernel_param[:, 1:3] = augmented
    return batch_kernel_param

def random_batch_kernel(batch, l=15, sig_min=0.25, sig_max=1.75, rate_iso=0.3, scaling=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def batch_kernel(batch, l=15, sigma=0.5, kernel_type=1, scaling=3, tensor=True):
    """
    generate definite kernel for test,
    :param kernel_type: 1 for iso gaussian, 2 for aniso, 3 for direct
    :param sigma: from 0.25 to 0.75
    """
    batch_kernel = np.zeros((batch, l, l))
    if kernel_type == isotropic_kernel_label:
        for i in range(batch):
            batch_kernel[i] = isotropic_gaussian_kernel(l=l, sigma=sigma, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def batch_kernel_param(batch, l=15, sigma=0.5, kernel_type=1, scaling=3, tensor=True):
    """
    Generate definite kernel for test,
    :param kernel_type: 1 for iso gaussian, 2 for aniso, 3 for direct
    :param sigma: from 0.25 to 0.75
    :returns kernel, kernel_param
    """
    batch_kernel = np.zeros((batch, l, l))
    param = np.zeros((batch, 3))
    param[:, 0] = sigma
    param[:, 1] = sigma
    for i in range(batch):
        batch_kernel[i] = isotropic_gaussian_kernel(l=l, sigma=sigma, tensor=False)
    return (torch.FloatTensor(batch_kernel), torch.FloatTensor(param)) if tensor else (batch_kernel, param)


def batch_kernel_from_param(batch_kernel_param, l=15, cuda=False, tensor=True):
    """
    generate Gaussian Blur Kernel from Param(sig_x, sig_y, pi)
    :param batch_kernel_param:
    :param l:
    :param tensor:
    :return:
    """
    batch_kernel = np.zeros((batch_kernel_param.shape[0], l, l))
    for i in range(batch_kernel_param.shape[0]):
        # kernel_type, x, y, pi = batch_kernel_param[i]
        x, y, pi = batch_kernel_param[i]
        kernel_type = 0 if x == y else 1
        # kernel_type = round(kernel_type.data[0]) if isinstance(kernel_type, Variable) else round(kernel_type)
        if kernel_type == isotropic_kernel_label:
            batch_kernel[i] = isotropic_gaussian_kernel(l=l, sigma=x, tensor=False)
        elif kernel_type == anisotropic_kernel_label:
            sig = cal_sigma(x, y, pi)
            batch_kernel[i] = anisotropic_gaussian_kernel(l=l, sigma_matrix=sig, tensor=False)
        else:
            sig = cal_sigma(x, y, pi)
            batch_kernel[i] = anisotropic_gaussian_kernel(l=l, sigma=x, tensor=False)

    if tensor:
        return torch.FloatTensor(batch_kernel).cuda() if cuda else torch.FloatTensor(batch_kernel)
    else:
        return batch_kernel

def random_batch_gaussian_noise_param(im, sig_min=0, sig_max=0.15, mean=0):
    """
    Random level Additive Gaussian White Noise
    :param im:
    :param sig_min:
    :param sig_max:
    :param mean:
    :return:
    """
    B, C, H, W = im.shape
    means = torch.zeros(B, C, H, W)
    std = torch.ones(B, C, H, W)
    noise_param = torch.rand(B) * (sig_max - sig_min) + sig_min
    rand_sig = noise_param.view(B, 1, 1, 1).expand(B, C, H, W)
    sigma = rand_sig * std
    noise = torch.normal(means, sigma)
    return noise, noise_param

def batch_gaussian_noise_param(im, sig=0.1, mean=0):
    """
    Sig level Additive Gaussian White Noise
    :param im:
    :param sig:
    :param mean:
    :return:
    """
    B, C, H, W = im.shape
    means = torch.zeros(B, C, H, W)
    std = torch.ones(B, C, H, W)
    noise_param = torch.ones(B) * sig
    rand_sig = noise_param.view(B, 1, 1, 1).expand(B, C, H, W)
    sigma = rand_sig * std
    noise = torch.normal(means, sigma)
    return noise, noise_param


### old version of gaussian kernels, used to be in RLSR
# def cal_sigma(sig_x, sig_y, radians):
#     D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
#     U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
#     sigma = np.dot(U, np.dot(D, U.T))
#     return sigma
#
#
# def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
#     ax = np.arange(-l // 2 + 1., l // 2 + 1.)
#     xx, yy = np.meshgrid(ax, ax)
#     xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
#     inverse_sigma = np.linalg.inv(sigma_matrix)
#     kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
#     return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
#
#
# def isotropic_gaussian_kernel(l, sigma, tensor=False):
#     ax = np.arange(-l // 2 + 1., l // 2 + 1.)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
#     return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
#
#
# def random_anisotropic_gaussian_kernel(sig_min=0.25, sig_max=1.75, scaling=3, l=15, tensor=False):
#     pi = np.random.random() * math.pi * 2 - math.pi
#     x = np.random.random() * (sig_max - sig_min) + sig_min
#     y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
#     sig = cal_sigma(x, y, pi)
#     k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
#     return k
#
#
# def random_isotropic_gaussian_kernel(sig_min=0.25, sig_max=1.75, l=15, tensor=False):
#     x = np.random.random() * (sig_max - sig_min) + sig_min
#     k = isotropic_gaussian_kernel(l, x, tensor=tensor)
#     return k
#
#
# def random_gaussian_kernel(l=15, sig_min=0.25, sig_max=1.75, rate_iso=0.3, scaling=3, tensor=False):
#     if np.random.random() < rate_iso:
#         return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
#     else:
#         return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)
#
#
# def random_batch_kernel(batch, l=15, sig_min=0.25, sig_max=1.75, rate_iso=0.3, scaling=3, tensor=True):
#     batch_kernel = np.zeros((batch, l, l))
#     for i in range(batch):
#         batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
#     return torch.FloatTensor(batch_kernel) if tensor else batch_kernel
#
#
# def batch_kernel(batch, l=15, sigma=0.5, kernel_type=1, scaling=3, tensor=True):
#     """
#     generate definite kernel for test,
#     :param kernel_type: 1 for iso gaussian, 2 for aniso, 3 for direct
#     :param sigma: from 0.25 to 0.75
#     """
#     batch_kernel = np.zeros((batch, l, l))
#     if kernel_type == 1:
#         for i in range(batch):
#             batch_kernel[i] = isotropic_gaussian_kernel(l=l, sigma=sigma, tensor=False)
#     # elif kernel_type == 2:
#     #     sigma_matrix
#     #     for i in range(batch):
#     #         batch_kernel[i] = anisotropic_gaussian_kernel(l=l, sigma_matrix, tensor=False)
#     return torch.FloatTensor(batch_kernel) if tensor else batch_kernel







