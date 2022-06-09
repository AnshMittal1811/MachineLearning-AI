import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import math
import os
from collections import OrderedDict
import torchvision

import numpy as np

import sys
sys.path.append('.')
from modules.msssim import msssim


def gradient2d(x, absolute=False, square=False):
    # x should be B x C x H x W
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]

    # zeros = tf.zeros_like(x)
    zeros = torch.zeros_like(x)
    zero_h = zeros[:, :, 0:1, :]
    zero_w = zeros[:, :, :, 0:1]
    dh = torch.cat([dh, zero_h], axis=2)
    dw = torch.cat([dw, zero_w], axis=3)
    if absolute:
        dh = torch.abs(dh)
        dw = torch.abs(dw)
    if square:
        dh = dh ** 2
        dw = dw ** 2
    return dh, dw

def smoothnessloss(feat, mask=None):
    # mask is B x 1 x H x W
    dh, dw = gradient2d(feat, absolute=True)
    smooth_pix = torch.mean(dh+dw, dim=1, keepdims=True) # B x 1 x H x W
    if mask is not None:
        smooth_loss = torch.sum(smooth_pix * mask) / torch.sum(mask)
    else:
        smooth_loss = torch.mean(smooth_pix)

    return smooth_loss

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, padding=(kernel_size-1)//2, padding_mode='replicate',
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def compute_ssim(img1, img2):
    # img1, img2: [0, 255]

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(-1)
        
def fetch_optimizer(args, model):
    # todo: enable the dict
    """ Create the optimizer and learning rate scheduler """
    special_args_dict = args.special_args_dict

    named_params = dict(model.named_parameters())
    default_params = [x[1] for x in named_params.items() if x[0] not in list(special_args_dict.keys())]

    param_group_lr_list = [args.lr]
    special_params_list = [{'params': default_params, 'lr': args.lr}, ] # use the default lr (args.lr)

    for (name, lr) in special_args_dict.items():
        if name in named_params.keys():
            special_params_list.append({'params': named_params[name], 'lr': lr})
            param_group_lr_list.append(lr)
        else:
            print('warning: param key %s does not exist in model' % name)

    optimizer = optim.AdamW(special_params_list, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, param_group_lr_list, args.num_steps + 100,
        pct_start=args.pct_start, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def sequence_loss_rgb(rgb_est,
                        rgb_gt,
                        mask_gt=None,
                        loss_type='l1',
                        lpips_vgg=None,
                        weight=None,
                        gradual_weight=None,
                        gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(rgb_est)
    flow_loss = 0.0

    ht, wd = rgb_gt.shape[-2:]

    if mask_gt is None:
        mask_gt = torch.ones_like(rgb_gt)
    else:
        mask_gt = mask_gt.expand_as(rgb_gt)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        if loss_type == "l1":
            i_loss = (rgb_est[i]*mask_gt - rgb_gt*mask_gt).abs()
        elif loss_type == "l2" or loss_type == "linear_l2":
            i_loss = (rgb_est[i]*mask_gt - rgb_gt*mask_gt)**2
        elif loss_type == "msssim":
            i_loss = 1. - msssim(rgb_est[i]*mask_gt, rgb_gt*mask_gt, val_range=2, normalize="relu")
            # i_loss = ssim(rgb_est[i] * mask_gt, rgb_gt * mask_gt, val_range=2)
        elif loss_type == "mix_l1_msssim":
            msssim_loss = 1. - msssim(rgb_est[i]*mask_gt, rgb_gt*mask_gt, val_range=2, normalize="relu")
            l1_loss = (rgb_est[i]*mask_gt - rgb_gt*mask_gt).abs()
            # alpha value from this paper https://arxiv.org/pdf/1511.08861
            alpha = 0.84
            i_loss = alpha * msssim_loss + (1.-alpha) * l1_loss
        else:
            raise NotImplementedError

        if not weight is None:
            i_loss *= weight

        # flow_loss += i_weight * i_loss.mean()
        flow_loss += i_weight * i_loss.sum() / mask_gt.sum()

    rgb_gt_scaled = (rgb_gt + 1.0) / 2.0 # range [0,1]
    rgb_est_scaled = (rgb_est[-1].detach() + 1.0) / 2.0

    l1 = (rgb_est_scaled * mask_gt - rgb_gt_scaled * mask_gt).abs().sum() / torch.sum(mask_gt) # proper normalize, consider the size of the mask
    l2 = ((rgb_est_scaled * mask_gt - rgb_gt_scaled * mask_gt)**2).sum() / torch.sum(mask_gt)
    psnr = -10. * torch.log(l2) / np.log(10.0)

    ssim = 0.0

    B = rgb_gt_scaled.shape[0]
    for b in range(B):
        g = (rgb_gt_scaled * mask_gt)[b].permute(1,2,0).cpu().numpy()
        p = (rgb_est_scaled * mask_gt)[b].permute(1,2,0).cpu().numpy()
        m = mask_gt[b].permute(1,2,0).cpu().numpy()

        ssim_custom = compute_ssim(g*255.0, p*255.0)
        ssim_custom = np.mean(ssim_custom, axis=2)
        ssim_custom = np.pad(ssim_custom, ((5, 5), (5, 5)), mode='mean')
        ssim_custom = ssim_custom[..., None]
        ssim += np.sum(ssim_custom * m) / np.sum(m)

        # sanity check: same when no mask
        # print(np.sum(ssim_custom * m) / np.sum(m))
        # print(np.mean(compute_ssim(g*255.0, p*255.0)))

        # ssim += structural_similarity(g, p, data_range=1.0, multichannel=True)

    ssim = ssim / float(B)

    assert (not torch.isnan(rgb_est[-1]).any())
    assert (not torch.isinf(rgb_est[-1]).any())

    metrics = {
        'l1': l1.item(),
        'l2': l2.item(),
        'psnr': psnr.item(),
        'ssim': ssim,
    }

    if lpips_vgg is not None:
        with torch.no_grad():
            # input should have range [-1,1], which we already have; need bgr2rgb
            # have sanity check that psnr, ssim and lpips is exactly the same as the tf version https://github.com/bmild/nerf/issues/66
            lpips_val = lpips_vgg(rgb_gt[:, [2,1,0]], rgb_est[-1][:, [2,1,0]])
            lpips_val = lpips_val.mean().cpu().item()
            metrics['lpips'] = lpips_val

    # print(metrics)

    return flow_loss, metrics