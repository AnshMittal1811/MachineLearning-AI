# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque
from typing import Dict

import torch


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def values(self):
        d = torch.tensor(list(self.series))
        return d

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def global_median(self):
        d = torch.tensor(list(self.series))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def min(self):
        d = torch.tensor(list(self.series))
        return d.min().item()

    @property
    def max(self):
        d = torch.tensor(list(self.series))
        return d.max().item()

    @property
    def summary(self):
        if len(self.series) == 1:
            return self.series[0]
        elif len(self.series) == 0:
            return 0.0
        else:
            return {"min": self.min,
                    "max": self.max,
                    "avg": self.global_avg,
                    "median": self.global_median}


class MetricLogger:
    def __init__(self, delimiter="\t", window_size=20):
        self.meters: Dict[SmoothedValue] = defaultdict(lambda: SmoothedValue(window_size))
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __contains__(self, attr):
        return attr in self.meters

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def get(self, attr):
        return self.__getattr__(attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg))
        return self.delimiter.join(loss_str)

    def summary(self):
        return {name: meter.summary for name, meter in self.meters.items()}

    def raw(self):
        return {name: meter.values for name, meter in self.meters.items()}
