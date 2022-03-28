"""
This file contains the utility functions needed for GANs.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

import numpy as np
from torch import Tensor, from_numpy, randn, full
import torch.nn as nn
from torch.autograd.variable import Variable


def images_to_vectors(images: Tensor) -> Tensor:
    """ converts (Nx28x28) tensor to (Nx784) torch tensor """
    return images.view(images.size(0), 32 * 32)


def images_to_vectors_numpy(images: np.array) -> Tensor:
    """ converts (Nx28x28) np array to (Nx784) torch tensor """
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2], images.shape[3])
    return from_numpy(images[:, :, 0])


def images_to_vectors_numpy_multiclass(images: np.array) -> Tensor:
    """ converts (Nx28x28) numpy array to (Nx784) tensor in multiclass setting"""
    images = images.reshape(images.shape[0], images.shape[2]*images.shape[3], images.shape[1])
    return from_numpy(images[:, :, 0])


def vectors_to_images_numpy(vectors: np.array) -> np.array:
    """ converts (Nx784) tensor to (Nx28x28) numpy array """
    return vectors.reshape(vectors.shape[0], 32, 32)


def vectors_to_images(vectors):
    """ converts (Nx784) tensor to (Nx32x32) tensor """
    return vectors.view(vectors.size(0), 1, 32, 32)


def vectors_to_images_cifar(vectors):
    """ converts (Nx784) tensor to (Nx32x32) tensor """
    return vectors.view(vectors.size(0), 3, 32, 32)


def noise(size: int, cuda: False) -> Variable:
    """ generates a 1-d vector of normal sampled random values of mean 0 and standard deviation 1 """
    result = Variable(randn(size, 100))
    if cuda:
        result = result.cuda()
    return result


def noise_cifar(size: int, cuda: False) -> Variable:
    """ generates a 1-d vector of normal sampled random values of mean 0 and standard deviation 1"""
    result = Variable(randn(size, 100, 1, 1))
    if cuda:
        result = result.cuda()
    return result


def values_target(size: tuple, value: float, cuda: False) -> Variable:
    """ returns tensor filled with value of given size """
    result = Variable(full(size=size, fill_value=value))
    if cuda:
        result = result.cuda()
    return result


def weights_init(m):
    """ initialize convolutional and batch norm layers in generator and discriminator """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
