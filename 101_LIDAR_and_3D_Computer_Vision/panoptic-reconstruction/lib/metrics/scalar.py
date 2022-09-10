from typing import Union, Dict

import torch

from lib.metrics import Metric


class Scalar(Metric):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.values = []
        self._cached: torch.Tensor = torch.zeros(1)
        self._is_invalid = True
        self.reduction = reduction

    def reduce(self) -> Union[float, Dict[str, float]]:
        if self._is_invalid:
            self._cached = torch.tensor(self.values)
            self._is_invalid = False

        if self.reduction == "mean":
            return self.reduce_mean()
        elif self.reduction == "median":
            return self.reduce_median()
        elif self.reduction == "summary":
            return self.reduce_summary()

    def reduce_mean(self) -> float:
        mean = self._cached.mean().item()
        return mean

    def reduce_median(self) -> float:
        median = self._cached.median().item()
        return median

    def reduce_summary(self) -> Dict[str, float]:
        if self._cached.shape[0] == 0:
            summary = {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
            }
        else:
            summary = {
                "min": self._cached.min().item(),
                "max": self._cached.max().item(),
                "mean": self.reduce_mean(),
                "median": self.reduce_median(),
            }

        return summary
