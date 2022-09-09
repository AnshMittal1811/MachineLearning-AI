import numpy as np
import torch
from scipy import signal
import lpips


def mse(img0, img1, mask=None):
    """MSE error"""
    if mask is None:
        return np.mean((img0 - img1) ** 2)

    masked_se = (mask * (img0 - img1)) ** 2
    return np.sum(masked_se) / np.sum(mask)


def psnr(img0, img1, mask=None, max_val=255):
    """Takes in two numpy images in range [0, max_val]."""
    img0 = img0 / max_val
    img1 = img1 / max_val
    return -10 * np.log10(mse(img0, img1, mask=mask))


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def ssim(img1, img2, max_val=255, filter_size=11,
         filter_sigma=1.5, k1=0.01, k2=0.03, mask=None):
    """
    Original code here: https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/compression/image_encoder/msssim.py
    Return the Structural Similarity Map between `img1` and `img2`.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
            maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
            for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
            the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
            the original paper).
    Returns:
        Pair containing the mean SSIM and contrast sensitivity between `img1` and
        `img2`.
    Raises:
        RuntimeError: If input images don't have the same shape or don't have four
            dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError("Input images must have the same shape (%s vs. %s).",
                           img1.shape, img2.shape)
    if img1.ndim == 3:
        img1 = np.expand_dims(img1, 0)

    if img2.ndim == 3:
        img2 = np.expand_dims(img2, 0)

    if img1.ndim != 4:
        raise RuntimeError(
            "Input images must have four dimensions, not %d", img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(fspecial_gauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode="same")
        mu2 = signal.fftconvolve(img2, window, mode="same")
        sigma11 = signal.fftconvolve(img1 * img1, window, mode="same")
        sigma22 = signal.fftconvolve(img2 * img2, window, mode="same")
        sigma12 = signal.fftconvolve(img1 * img2, window, mode="same")
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    if mask is not None:
        score = (((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2))
        score = np.sum(mask * score) / (np.sum(mask * np.ones_like(score)))
    else:
        score = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    # cs = np.mean(v1 / v2)
    return score


class LPIPSMetric:
    def __init__(self, device, spatial=False):
        # Initializing the model
        self.device = device
        self.spatial = spatial
        self.lpips_loss = lpips.LPIPS(net='alex', spatial=spatial).to(device)
        self.lpips_loss.eval()

    def eval_single_pair(self, img0, img1, max_val=255):
        # Load images
        img0 = lpips.im2tensor(img0, factor=max_val / 2.).to(self.device)
        img1 = lpips.im2tensor(img1, factor=max_val / 2.).to(self.device)

        # Compute distance
        with torch.no_grad():
            if self.spatial:
                error = self.lpips_loss.forward(img0, img1).cpu().numpy()[0, 0]
            else:
                error = self.lpips_loss.forward(img0, img1).cpu().item()
        return error

    def eval_single_pair_masked(self, img0, img1, mask, max_val=255):
        assert self.spatial, "spatial needs to be set True to run masked LPIPS."
        assert len(mask.shape) == 2, f"expected mask to be a 2D array, but got shape {mask.shape}"
        lpips_map = self.eval_single_pair(img0, img1, max_val=max_val)
        mask = mask.astype(lpips_map.dtype)
        return (lpips_map * mask).sum() / mask.sum()

    def close(self):
        self.lpips_loss.to('cpu')
        del self.lpips_loss
