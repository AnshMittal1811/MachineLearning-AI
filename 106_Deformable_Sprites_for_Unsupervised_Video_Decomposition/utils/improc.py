import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def pad_diff(x1, x2):
    """
    pads x1 so that shape matches x2
    :param x1 (*, H1, W1), x2 (*, H2, W2)
    :return x1 padded (*, H2, W2)
    """
    diffY = x2.size()[-2] - x1.size()[-2]
    diffX = x2.size()[-1] - x1.size()[-1]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return x1


def get_gaussian_pyr(img, n_levels, ksize=3, sigma=1):
    cur = img
    pyr = [img]
    for _ in range(n_levels):
        cur = TF.gaussian_blur(cur, kernel_size=ksize, sigma=sigma)
        cur = cur[..., ::2, ::2]
        pyr.append(cur)
    return pyr[::-1]


def get_laplacian_pyr(img, n_levels, ksize=3, sigma=1):
    # downscale n_levels + 1, ..., 0
    gp = get_gaussian_pyr(img, n_levels + 1, ksize, sigma)
    pyr = []
    for i in range(n_levels):
        reup = F.interpolate(
            gp[i], scale_factor=2, mode="bilinear", align_corners=False
        )
        reup = pad_diff(reup, gp[i + 1])
        pyr.append(gp[i + 1] - reup)
    return pyr


def get_edge_kernel(ksize, device=None):
    if device is None:
        device = torch.device("cpu")
    kxy, kxx = cv2.getDerivKernels(1, 0, ksize)  # x gradients
    kx = (kxx * kxy.T).reshape(1, 1, ksize, ksize)
    kyy, kyx = cv2.getDerivKernels(0, 1, ksize)  # y gradients
    ky = (kyx * kyy.T).reshape(1, 1, ksize, ksize)
    kernel = np.concatenate([kx, ky], axis=0)  # (2, 1, ksize, ksize)
    return torch.tensor(kernel, device=device, dtype=torch.float32)


def get_edges(src, ksize=3, kernel=None):
    """
    get conv2d operator that will compute the x and y spatial gradients (Sobel)
    for each channel will apply both x and y gradients
    will apply each gradient filter per channel, (N, C, H, W) -> (N, 2*C, H, W)
    """
    if kernel is None:
        kernel = get_edge_kernel(ksize, device=src.device)
    K = kernel.shape[-1]
    C = src.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)  # (2 * C, 1, K, K)
    padding = (K - 1) // 2
    return F.conv2d(src, kernel, padding=padding, groups=C)  # (N, 2*C, *)


def get_edge_map(src, ksize=3, norm=1, kernel=None):
    edges = get_edges(src, ksize, kernel)
    return torch.linalg.norm(edges, ord=norm, dim=1, keepdim=True)  # (N, 1, *)


def alpha_to_weights(alpha):
    """
    Compute the weights using the alpha layers
    W_1 = A_1
    W_2 = (1 - A_1) * A_2
        ...
    W_M = (1 - A_1) * ... * (1 - A_{M-1}) * 1
    :param alpha (N, M-1, *)
    """
    alpha_bg = torch.cat([alpha, torch.ones_like(alpha[:, :1])], dim=1)  # (N, M, *)
    alpha_shifted = torch.cat(
        [torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], dim=1
    )  # (N, M, *) 1, 1 - A1, ..., 1 - A_{M-1}
    T = torch.cumprod(alpha_shifted, dim=1)  # (N, M, *)
    return T * alpha_bg  # (N, M, *)


def weights_to_alpha(weights):
    """
    :param weights (N, M, *)
    """
    weights_shifted = torch.cat([torch.zeros_like(weights[:, 0:1]), weights], dim=1)
    P = torch.cumsum(weights_shifted, dim=1)  # (N, M+1, *)
    return weights / (1 - P[:, :-1] + 1e-5)  # (N, M, *)


def tensor2image(tensor):
    """:param tensor (C, H, W)"""
    return tensor.detach().permute(1, 2, 0).cpu().numpy()


def compute_psnr(pred, tgt):
    """
    :param pred (*, C, H, W)
    :param tgt (*, C, H, W)
    """
    mse = torch.mean((pred - tgt) ** 2, dim=(-1, -2, -3))
    psnr = -10 * torch.log10(mse)
    return psnr


def compute_iou(pred, target, dim=None):
    """
    Compute region similarity as the Jaccard Index.
    :param pred (binary tensor) prediction
    :param target (binary tensor) ground truth
    :param dim (optional, int) the dimension to reduce across
    :returns jaccard (float) region similarity
    """
    intersect = pred & target
    union = pred | target
    if dim is None:
        intersect = intersect.sum()
        union = union.sum()
    else:
        intersect = intersect.sum(dim)
        union = union.sum(dim)
    return (intersect + 1e-6) / (union + 1e-6)
