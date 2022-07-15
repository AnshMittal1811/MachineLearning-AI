import torch
import torch.nn as nn


def group_weight(module):
    # Group module parameters into two group
    # One need weight_decay and the other doesn't
    # Copy from
    # https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/train.py
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]


def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        scale_running_lr = max((1. - frac), 0.) ** args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr


class Statistic():
    '''
    For training statistic
    set winsz > 0 for running statitic
    '''
    def __init__(self, winsz=0):
        self.winsz = winsz
        self.cnt = 0
        self.weight = 0  # work only if winsz==0
        self.total = 0

    def update(self, val, weight=1):
        self.cnt += 1
        if self.winsz <= 0:
            self.weight += weight
            self.total += val * weight
        elif self.cnt > self.winsz:
            self.total += (val - self.total) / self.winsz
        else:
            self.total += (val - self.total) / self.cnt

    def __str__(self):
        return '%.6f' % float(self)

    def __float__(self):
        if self.winsz <= 0:
            return float(self.total / self.weight)
        else:
            return float(self.total)


class StatisticDict():
    '''
    Wrapper for Statistic
    '''
    def __init__(self, winsz=0):
        self.winsz = winsz
        self._map = {}
        self._order = []

    def update(self, k, val, weight=1):
        if k not in self._map:
            self._map[k] = Statistic(self.winsz)
            self._order.append(k)
        self._map[k].update(val, weight)

    def __str__(self):
        return ' | '.join([
            '%s %.6f' % (k, self._map[k]) for k in self._order])


def pmap_x(pmap1, pmap2):
    pmap_max = torch.max(pmap1 + 1e-9, pmap2 + 1.1e-9)
    pmap_min = torch.min(pmap1 + 1e-9, pmap2 + 1.1e-9)
    return pmap_min / pmap_max
