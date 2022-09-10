import torch
from typing import Dict, Union, Tuple

from lib.metrics import Metric, Scalar


class Representation:
    def __init__(self):
        pass

    def __call__(self, volume: torch.Tensor):
        return volume


class Occupancy(Representation):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, volume: torch.Tensor) -> torch.BoolTensor:
        binary: torch.BoolTensor = (volume > self.threshold).bool()

        return binary


class DistanceField(Representation):
    def __init__(self, threshold: float):
        super().__init__()

        self.threshold = threshold

    def __call__(self, volume: torch.Tensor) -> torch.BoolTensor:
        binary: torch.BoolTensor = (volume < self.threshold).bool()

        return binary


class SignedDistanceField(Representation):
    def __init__(self, known_threshold: float, unknown_threshold: float):
        super().__init__()
        self.known_threshold = known_threshold
        self.unknown_threshold = unknown_threshold

    def __call__(self, volume: torch.Tensor) -> torch.BoolTensor:
        binary_known: torch.BoolTensor = (volume < self.known_threshold).bool()
        binary_unknown: torch.BoolTensor = (volume > self.unknown_threshold).bool()
        binary = binary_known & binary_unknown
        return binary


class IntersectionOverUnion(Scalar):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def add(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> None:
        iou = compute_iou(prediction, ground_truth)
        self.values.append(iou)
        self._is_invalid = True


def intersection(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (ground_truth.bool() & prediction.bool()).float()


def union(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (ground_truth.bool() | prediction.bool()).float()


def difference(ground_truth: torch.Tensor, prediction: torch.Tensor,
               two_sided=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    diff = (ground_truth.int() - prediction.int()).float()

    if not two_sided:
        return torch.abs(diff)
    else:
        false_positive = (diff < 0).float()  # GT: 0, pred: 1, diff: -1 --> false positive
        false_negative = (diff > 0).float()  # GT: 1, pred: 0, diff: 1 --> false negative

        return false_positive, false_negative


def compute_iou(ground_truth: torch.Tensor, prediction: torch.Tensor) -> float:
    num_intersection = float(torch.sum(intersection(ground_truth, prediction)))
    num_union = float(torch.sum(union(ground_truth, prediction)))
    iou = 0.0 if num_union == 0 else num_intersection / num_union
    return iou
