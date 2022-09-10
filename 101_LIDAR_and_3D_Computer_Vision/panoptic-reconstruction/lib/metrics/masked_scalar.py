from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, List

import torch

from lib.metrics import Metric


class MaskedScalar(Metric):
    def __init__(self, reduction: str = "mean", ignore_labels: List[int] = None):
        super().__init__()
        self.values = defaultdict(list)
        self.totals = defaultdict(int)

        self.reduction = reduction

        if ignore_labels is None:
            self.ignore_labels = []
        else:
            self.ignore_labels = ignore_labels

        self._cached: Dict[str, torch.Tensor] = {}
        self._is_valid = True

    def add(self, prediction: torch.Tensor, ground_truth: Tuple[torch.Tensor, Dict[int, torch.Tensor]]):
        ground_truth, masks = ground_truth

        for label, mask in masks.items():
            if not self._should_label_be_ignored(label):
                masked_ground_truth = torch.masked_select(ground_truth, mask)
                masked_prediction = torch.masked_select(prediction, mask)
                scalar = self.evaluate_sample(masked_ground_truth, masked_prediction)
                self.values[label].append(scalar)
                self.totals[label] += masked_prediction.numel()

        self._is_valid = False

    def evaluate_sample(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
        error = 0
        return error

    def _cache(self) -> None:
        if not self._is_valid:
            self._cached.clear()
            for label, values in self.values.items():
                self._cached[label] = torch.tensor(values)

            self._is_valid = True

    def reduce(self):
        self._cache()

        if self.reduction == "mean":
            return self._reduce_mean()[0]
        if self.reduction == "instance_mean":
            return self._reduce_instance_mean()
        elif self.reduction == "summary":
            return self._reduce_summary()

    def _reduce_mean(self):
        per_label = OrderedDict()

        sorted_by_labels = sorted(list(self._cached.items()), key=lambda x: x[0])

        for label, values in sorted_by_labels:
            per_label[label] = values.mean().item()

        mean = torch.mean(torch.tensor(list(per_label.values()))).item()

        return mean, per_label

    def _reduce_instance_mean(self):
        if self._cached and len(self._cached) > 0:
            values = list(self._cached.values())

            if values[0].dim() == 0:
                values = torch.tensor(values)
            else:
                values = torch.cat(values)
            mean = torch.mean(values).item()
        else:
            mean = 0

        return mean

    def _reduce_summary(self):
        mean, per_label = self._reduce_mean()

        summary = OrderedDict()
        summary["mean"] = mean
        summary.update(per_label)

        return summary

    def _should_label_be_ignored(self, label):
        if label in self.ignore_labels:
            return True
        else:
            return False
