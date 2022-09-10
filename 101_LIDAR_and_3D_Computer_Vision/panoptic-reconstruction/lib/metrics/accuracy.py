from collections import defaultdict, OrderedDict

from typing import Dict, Tuple

from lib.metrics import *
import torch


class Accuracy(Metric):
    def __init__(self, reduction: str = "instance"):
        super().__init__()
        self.corrects = defaultdict(int)
        self.totals = defaultdict(int)
        self.reduction = reduction

    def add(self, prediction, ground_truth):
        for pred, gt in zip(prediction, ground_truth):
            if pred == gt:
                self.corrects[gt] += 1

            self.totals[gt] += 1

    def reduce(self):
        if self.reduction == "instance":
            return self.reduce_instance()
        elif self.reduction == "class":
            class_accuracy, _ = self.reduce_class()
            return class_accuracy
        elif self.reduction == "summary":
            return self.reduce_summary()

    def reduce_instance(self):
        total = 0
        correct = 0
        for label, num_total in self.totals.items():
            num_correct = self.corrects[label]
            total += num_total
            correct += num_correct

        accuracy = correct / total if total > 0 else 0.0

        return accuracy

    def reduce_class(self) -> Tuple[float, Dict]:
        per_class_accuracy = OrderedDict()

        sorted_totals = sorted(list(self.totals.items()), key=lambda x: x[0])

        for label, num_total in sorted_totals:
            num_correct = self.corrects[label]
            per_class_accuracy[label] = num_correct / num_total if num_total > 0 else 0.0

        mean = torch.mean(torch.tensor(list(per_class_accuracy.values()), dtype=torch.float)).item()

        return mean, per_class_accuracy

    def reduce_summary(self):
        mean, per_class = self.reduce_class()

        return {"mean": mean, **per_class}

    def reset(self):
        self.corrects.clear()
        self.totals.clear()
