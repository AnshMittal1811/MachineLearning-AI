# Torch version of flow visualization from https://github.com/tomrunia/OpticalFlow_Visualization

import torch
import numpy as np


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(uv, convert_to_bgr=False, to_uint8=False):
    """
    :param uv (*, 2, H, W) rescaled flow in x, y
    :param convert_to_bgr (bool, optional) default False
    returns flow image (*, 3, H, W)
    """
    *dims, _, H, W = uv.shape
    u, v = uv[..., 0, :, :], uv[..., 1, :, :]
    dtype = torch.uint8 if to_uint8 else torch.float32
    flow_image = torch.zeros(*dims, 3, H, W, dtype=dtype, device=u.device)
    colorwheel = torch.tensor(make_colorwheel(), device=u.device)  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = torch.sqrt(torch.square(u) + torch.square(v))
    a = torch.atan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = (tmp[k0.flatten()] / 255.0).reshape(k0.shape)
        col1 = (tmp[k1.flatten()] / 255.0).reshape(k1.shape)
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        chan = (255 * col).byte() if to_uint8 else col
        flow_image[..., ch_idx, :, :] = chan
    return flow_image


def normalize_flow(flow_uv, epsilon=1e-6):
    """
    :param flow_uv (*, 2, H, W) in normalized pixel coords [-1, 1]
    :return the max per uv_map, (*, 1, 1) with extra dims to match H, W dims
    """
    *dims, _, H, W = flow_uv.shape
    scale = torch.tensor([W / 2, H / 2], device=flow_uv.device)
    uv = flow_uv * scale.view(*(1,) * len(dims), 2, 1, 1)

    rad = torch.linalg.norm(uv, dim=-3, keepdim=True)  # (*, 1, H, W)
    rad_max = rad.amax(dim=(-1, -2), keepdims=True)  # (*, 1, 1, 1)
    flow_norm = rad_max + epsilon
    return uv / flow_norm, flow_norm


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, to_uint8=False):
    """
    :param flow_uv (*, 2, H, W) in normalized pixel coords (pixel ranges [-1, 1])
    :param clip_flow (float, optional): max clip of flow values. Defaults to None.
    :param convert_to_bgr (bool, optional) default False
    :returns (*, 3, H, W) color image of flow
    """
    if clip_flow is not None:
        flow_uv = torch.clamp_max(flow_uv, clip_flow)
    uv, flow_norm = normalize_flow(flow_uv)  # (*, 1, 1)
    return flow_uv_to_colors(uv, convert_to_bgr=convert_to_bgr, to_uint8=to_uint8)
