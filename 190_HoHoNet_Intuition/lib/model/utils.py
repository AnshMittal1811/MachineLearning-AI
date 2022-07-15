import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

import scipy
import numpy as np
from scipy.ndimage.filters import maximum_filter
from sklearn.linear_model import HuberRegressor


''' Panorama patch for layers '''
def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=-1)

class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)

def wrap_lr_pad(net):
    for name, m in net.named_modules():
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        if isinstance(m, nn.Conv2d):
            if m.padding[1] == 0:
                continue
            w_pad = int(m.padding[1])
            m.padding = (m.padding[0], 0)
            setattr(
                root, names[-1],
                nn.Sequential(LR_PAD(w_pad), m)
            )
        elif isinstance(m, nn.Conv1d):
            if m.padding == (0, ):
                continue
            w_pad = int(m.padding[0])
            m.padding = (0,)
            setattr(
                root, names[-1],
                nn.Sequential(LR_PAD(w_pad), m)
            )

def pano_upsample_w(x, s):
    if len(x.shape) == 3:
        mode = 'linear'
        scale_factor = s
    elif len(x.shape) == 4:
        mode = 'bilinear'
        scale_factor = (1, s)
    else:
        raise NotImplementedError
    x = torch.cat([x[...,-1:], x, x[...,:1]], dim=-1)
    x = F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
    x = x[...,s:-s]
    return x

class PanoUpsampleW(nn.Module):
    def __init__(self, s):
        super(PanoUpsampleW, self).__init__()
        self.s = s

    def forward(self, x):
        return pano_upsample_w(x, self.s)


''' Testing augmentation helper '''
def augment(x, flip, rotate, rotate_flip):
    aug_type = ['']
    x_augmented = [x]
    if flip:
        aug_type.append('flip')
        x_augmented.append(x.flip(dims=(-1,)))
    for shift in rotate:
        aug_type.append('rotate %d' % shift)
        x_augmented.append(x.roll(shifts=shift, dims=-1))
        if rotate_flip:
            aug_type.append('rotate_flip %d' % shift)
            x_augmented.append(x_augmented[-1].flip(dims=(-1,)))
    return torch.cat(x_augmented, 0), aug_type

def augment_undo(pred_augmented, aug_type):
    pred_augmented = pred_augmented.cpu().numpy()
    assert len(pred_augmented) == len(aug_type), 'Unable to recover testing aug'
    pred_final = 0
    for pred, aug in zip(pred_augmented, aug_type):
        if aug == 'flip':
            pred_final += np.flip(pred, axis=-1)
        elif aug.startswith('rotate'):
            if 'flip' in aug:
                pred = np.flip(pred, axis=-1)
            shift = int(aug.split()[-1])
            pred_final += np.roll(pred, -shift, axis=-1)
        elif aug == '':
            pred_final += pred
        else:
            raise NotImplementedError

    return pred_final / len(aug_type)


''' Post-processing '''
def peaks_mask_torch(x1d, winsz=7, min_v=0.5):
    pad = winsz // 2
    x1d_max = F.max_pool1d(torch.cat([x1d[...,-pad:], x1d, x1d[...,:pad]], -1), winsz, stride=1)
    return (x1d == x1d_max) & (x1d >= min_v)

def peaks_finding_torch(x1d, winsz=7, min_v=0.5):
    ''' x1d: [B, 1, W] '''
    bid, _, cid = torch.where(peaks_mask_torch(x1d, winsz, min_v))
    return bid, cid

def peaks_finding(signal, winsz=7, min_v=0.5):
    max_v = maximum_filter(signal, size=winsz, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    return pk_loc
