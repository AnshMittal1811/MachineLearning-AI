import torch
from typing import Tuple, Dict, List

from lib.metrics.intersection_over_union import compute_iou
from lib.metrics import MaskedScalar


class SemanticIntersectionOverUnion(MaskedScalar):
    def __init__(self, reduction: str = "mean", ignore_labels: List[int] = None):
        super().__init__(reduction, ignore_labels)

    def add(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> None:
        ones = torch.ones([1], device=ground_truth.device)
        zeros = torch.zeros([1], device=ground_truth.device)

        labels = ground_truth.unique()

        for label in labels:
            if not self._should_label_be_ignored(label):

                ground_truth_mask = ground_truth == label
                prediction_mask = prediction == label

                masked_ground_truth = torch.where(ground_truth_mask, ones, zeros)
                masked_prediction = torch.where(prediction_mask, ones, zeros)
                scalar = self.evaluate_sample(masked_ground_truth, masked_prediction)
                self.values[label.item()].append(scalar)
                self.totals[label.item()] += masked_prediction.numel()  # for debug

        self._is_valid = False

    def evaluate_sample(self, ground_truth: torch.Tensor, prediction: torch.Tensor) -> float:
        iou = compute_iou(ground_truth, prediction)
        return iou
