import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision


def composite_rgba_checkers(masks, rgbs, n_rows=24, fac=0.2):
    """
    :param masks (*, 1, H, W)
    :param rgbs (*, 3, H, W)
    """
    *dims, _, H, W = masks.shape
    checkers = get_gray_checkerboard(H, W, n_rows, fac, device=masks.device)
    checkers = checkers.view(*(1,) * len(dims), 3, H, W)
    return masks * rgbs + (1 - masks) * checkers


def get_rainbow_checkerboard(H, W, n_rows, fac=0.2, device=None):
    checkers = get_checkerboard(H, W, n_rows, device=device)
    rainbow = get_rainbow(H, W, device=device)
    return fac * checkers + (1 - fac) * rainbow


def get_gray_checkerboard(H, W, n_rows, fac=0.2, shade=0.7, device=None):
    if device is None:
        device = torch.device("cpu")
    checkers = get_checkerboard(H, W, n_rows, device=device)
    bg = torch.ones(3, H, W, device=device, dtype=torch.float32) * shade
    return fac * checkers + (1 - fac) * bg


def get_checkerboard(H, W, n_rows, device=None):
    if device is None:
        device = torch.device("cpu")
    stride = H // n_rows
    n_cols = W // stride
    checkers = np.indices((n_rows, n_cols)).sum(axis=0) % 2
    checkers = cv2.resize(checkers, (W, H), interpolation=cv2.INTER_NEAREST)
    checkers = torch.tensor(checkers, device=device, dtype=torch.float32)
    return checkers[None].repeat(3, 1, 1)


def get_rainbow(H, W, device=None):
    if device is None:
        device = torch.device("cpu")
    uu, _ = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    cmap = plt.get_cmap("rainbow")
    rainbow = cmap(uu)[..., :3]
    return torch.tensor(rainbow, device=device, dtype=torch.float32).permute(2, 0, 1)


def get_grad_image(tensor):
    if tensor.grad is None:
        print("requested grad is not available")
        return None
    grad = tensor.grad.detach().cpu()  # (N, M, 1, H, W)
    sign_im, vmax = get_sign_image(grad)
    return sign_im, vmax, grad


def get_sign_image(tensor):
    """
    :param tensor (*, 1, H, W) image-like single-channel tensor
    """
    vmax = torch.abs(tensor).amax(dim=(-1, -2), keepdim=True)  # (*, 1, 1, 1)
    pos = torch.zeros_like(tensor)
    pos[tensor > 0] = tensor[tensor > 0]
    neg = torch.zeros_like(tensor)
    neg[tensor < 0] = -tensor[tensor < 0]
    sign_im = (
        torch.cat([pos, neg, torch.zeros_like(tensor)], dim=-3) / vmax
    )  # (*, 3, H, W)
    return sign_im, vmax
