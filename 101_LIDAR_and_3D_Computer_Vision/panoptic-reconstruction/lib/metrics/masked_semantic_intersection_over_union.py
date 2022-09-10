import torch
from typing import Tuple, Dict, List

from lib.metrics import MaskedScalar
from lib.metrics.intersection_over_union import compute_iou


class MaskedSemanticIntersectionOverUnion(MaskedScalar):
    def __init__(self, reduction: str = "mean", ignore_labels: List[int] = None):
        super().__init__(reduction, ignore_labels)

    def add(self, prediction: torch.Tensor, ground_truth: Tuple[torch.Tensor, torch.Tensor]) -> None:
        ones = torch.ones([1], device=prediction.device)
        zeros = torch.zeros([1], device=prediction.device)

        ground_truth, mask = ground_truth
        visible = mask == 1

        labels = ground_truth.unique()

        for label in labels:
            if not self._should_label_be_ignored(label):

                ground_truth_mask = (ground_truth == label) & visible
                prediction_mask = (prediction == label) & visible

                masked_ground_truth = torch.where(ground_truth_mask, ones, zeros)
                masked_prediction = torch.where(prediction_mask, ones, zeros)
                scalar = self.evaluate_sample(masked_ground_truth, masked_prediction)
                self.values[label].append(scalar)
                self.totals[label] += masked_prediction.numel()

        self._is_valid = False

    def evaluate_sample(self, ground_truth: torch.Tensor, prediction: torch.Tensor) -> float:
        iou = compute_iou(ground_truth, prediction)
        return iou
