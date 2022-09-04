from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch


def accuracy(y_pred, y_true):
    return (y_pred.round() == y_true).float().mean()


def precision(y_pred, y_true):
    return torch.mul(y_pred.round(), y_true).sum() / y_pred.round().sum()


def recall(y_pred, y_true):
    return torch.mul(y_pred.round(), y_true).sum() / y_true.sum()


if __name__ == '__main__':
    pred = torch.tensor([1.0, 0.12, 0.5, 0.9, 0, 1.0, 0.12, 0.5, 0.9, 0], dtype=torch.float)
    true = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float)

    print('Accuracy :', accuracy(pred, true))
    print('Precision :', precision(pred, true))
    print('Recall :', recall(pred, true))
