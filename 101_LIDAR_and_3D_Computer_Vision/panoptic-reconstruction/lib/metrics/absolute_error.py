import torch

from lib.metrics import Scalar


class AbsoluteError(Scalar):
    def __init__(self, ignore_value: float, reduction: str = "mean"):
        super().__init__(reduction)
        self.ignore_value = ignore_value

    def add(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> None:
        mask = None
        if self.ignore_value is not None:
            mask = ground_truth != self.ignore_value

        error = torch.abs(ground_truth - prediction)

        if mask is not None:
            error = error[mask]

        if mask.sum() > 0:
            error = error.mean()
            self.values.append(error)

        self._is_invalid = True
