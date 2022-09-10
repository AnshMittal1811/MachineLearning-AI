import torch
from typing import List

from lib.metrics import MaskedScalar


class MaskedAbsoluteError(MaskedScalar):
    def evaluate_sample(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
        error = torch.abs(ground_truth - prediction).mean()
        return error
