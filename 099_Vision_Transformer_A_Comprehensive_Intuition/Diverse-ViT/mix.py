""" Mixup and Cutmix
Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)
Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""
import numpy as np
import torch


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    onehot_y = torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)
    onehot_y = onehot_y.unsqueeze(2)
    return onehot_y.repeat(1,1,196)

def mixup_target(target, num_classes, lam, patch_lam, smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    
    mix_y = y1 * patch_lam + y2 * (1. - patch_lam)
    single_y = y1[:,:,0] * lam + y2[:,:,0] * (1. - lam)

    return mix_y.permute(0,2,1), single_y

def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)

    bbox_area = (yh - yl) * (xh - xl)
    lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])

    return yl, yh, xl, xh, lam

def patch_wise_lam(img_shape, yl, yh, xl, xh, device='cuda'):

    binary_mask = torch.ones(img_shape[-2:], device=device)
    binary_mask[yl:yh, xl:xh] = 0
    # kernal-size = 16*16
    # image-size = 224*224
    patch_wise_lam = torch.ones(14,14, device=device)
    for y_patch_idx in range(14):
        for x_patch_idx in range(14):
            ys = y_patch_idx*16
            xs = x_patch_idx*16
            patch_wise_lam[y_patch_idx, x_patch_idx] = torch.mean(binary_mask[ys:ys+16, xs:xs+16])
    return patch_wise_lam.flatten()

class Mixup_diversity:
    """ Mixup/Cutmix that applies different params to each element or whole batch
    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, 
                label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        self.mix_prob = prob
        self.switch_prob = switch_prob

        self.label_smoothing = label_smoothing
        self.num_classes = num_classes


    def _params_per_batch(self):
        use_cutmix = np.random.rand() < self.switch_prob
        lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
            np.random.beta(self.mixup_alpha, self.mixup_alpha)

        lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if use_cutmix:
            yl, yh, xl, xh, lam = rand_bbox(x.shape, lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh] #x=0, x_flip=1
            patch_lam = patch_wise_lam(x.shape, yl, yh, xl, xh)
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
            patch_lam = torch.ones(196, device='cuda')*lam
        return patch_lam, lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        patch_lam, lam = self._mix_batch(x)
        patch_target, target = mixup_target(target, self.num_classes, lam, patch_lam, self.label_smoothing)
        return x, patch_target, target

