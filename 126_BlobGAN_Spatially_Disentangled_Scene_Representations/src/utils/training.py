# Training utilities
import math
import random
from itertools import groupby
from numbers import Number
from typing import Dict, List

import torch
from torch import nn, Tensor, autograd

from .distributed import is_rank_zero


def make_noise(batch, latent_dim, n_noise):
    if n_noise == 1:
        return torch.randn(len(batch), latent_dim).type_as(batch)
    return torch.randn(n_noise, len(batch), latent_dim).type_as(batch).unbind(0)


def mixing_noise(batch, latent_dim, prob):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2)
    else:
        return [make_noise(batch, latent_dim, 1)]


ACCUM_WARN = False


def accumulate(model1, model2, decay=0.999):
    global ACCUM_WARN
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    if len(par1.keys() & par2.keys()) == 0:
        if is_rank_zero() and not ACCUM_WARN:
            print('Cannot accumulate, likely due to FSDP parameter flattening. Skipping.')
            ACCUM_WARN = True
        return
    device = next(model1.parameters()).device
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data.to(device), alpha=1 - decay)




def freeze(model: nn.Module, layers: List[str] = None):
    frozen = []
    for name, param in model.named_parameters():
        if layers is None or any(name.startswith(l) for l in layers):
            param.requires_grad = False
            frozen.append(name)
    if is_rank_zero():
        depth_two_params = [k for k, _ in groupby(
            ['.'.join(n.split('.')[:2]).replace('.weight', '').replace('.bias', '') for n in frozen])]
        print(f'Froze {len(frozen)} parameters - {depth_two_params} - for model of type {model.__class__.__name__}')


def requires_grad(model: nn.Module, requires: bool):
    for param in model.parameters():
        param.requires_grad = requires


def unfreeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def fill_module_uniform(module, range, blacklist=None):
    if blacklist is None: blacklist = []
    for n, p in module.named_parameters():
        if not any([b in n for b in blacklist]):
            nn.init.uniform_(p, -range, range)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_D_stats(key: str, scores: Tensor, gt: bool) -> Dict[str, Number]:
    acc = 100 * (scores > 0).sum() / len(scores)
    if not gt:
        acc = 100 - acc
    return {
        f'score_{key}': scores.mean(),
        f'acc_{key}': acc
    }


# Losses adapted from https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
def D_R1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def G_path_loss(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )

    if grad.ndim == 3:  # [N_batch x N_latent x D]
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    elif grad.ndim == 2:  # [N_batch x D]
        path_lengths = torch.sqrt(grad.pow(2).sum(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths.mean()
