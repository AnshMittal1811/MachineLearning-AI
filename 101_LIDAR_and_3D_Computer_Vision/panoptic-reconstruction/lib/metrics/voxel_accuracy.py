from collections import defaultdict

from typing import Dict, Tuple, List

from lib.metrics import *
import torch


class VoxelAccuracy(Accuracy):
    def __init__(self, excluded_labels: List[int], reduction: str = "instance"):
        super().__init__(reduction)
        self.corrects = defaultdict(int)
        self.totals = defaultdict(int)
        self.excluded_labels = excluded_labels

    def add(self, prediction, ground_truth):
        labels, totals = torch.unique(ground_truth, return_counts=True)

        for label, total in zip(labels, totals):
            # ignore certain labels
            if label in self.excluded_labels:
                continue

            label_mask = torch.eq(ground_truth, label)
            predicted = torch.masked_select(prediction, label_mask)
            correct = predicted == label
            num_correct = correct.sum()
            self.corrects[label.item()] += num_correct.item()
            self.totals[label.item()] += total.item()
