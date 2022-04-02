
"""Scheduler for training."""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class AnnealingStepLR(_LRScheduler):
    """Anenealing scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        mu_i (float, optional): Initial learning rate.
        mu_f (float, optional): Final learning rate.
        n (float, optional): Annealing steps.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, mu_i: float = 5e-4,
                 mu_f: float = 5e-5, n: float = 1.6e6) -> None:
        self.mu_i = mu_i
        self.mu_f = mu_f
        self.n = n

        super().__init__(optimizer)

    def get_lr(self):
        return [max(self.mu_f + (self.mu_i - self.mu_f) *
                    (1.0 - self.last_epoch / self.n), self.mu_f)
                for base_lr in self.base_lrs]


class Annealer:
    """Annealer for training.

    Args:
        init (float): Initial value.
        final (float): Final value.
        steps (int): Number of annealing steps.
    """

    def __init__(self, init: float, final: float, steps: int, **kwargs
                 ) -> None:

        self.init = init
        self.final = final
        self.steps = steps

        # Current value
        self.t = 0
        self.current = init

    def __iter__(self):
        return self

    def __next__(self) -> float:
        self.t += 1
        value = max(
            self.final + (self.init - self.final) * (1 - self.t / self.steps),
            self.final
        )
        self.current = value

        return value


class SigmaAnnealer:
    """Annealer for sigma.

    Args:
        init (float): Initial value.
        final (float): Final value.
        constant (float): Constant value for pre-train
        steps (int): Number of annealing steps.
        pretrain (int): Number of pre-training steps.
    """

    def __init__(self, init: float, final: float, constant: float, steps: int,
                 pretrain: int, **kwargs) -> None:

        self.init = init
        self.final = final
        self.constant = constant
        self.steps = steps
        self.pretrain = pretrain

        if steps < pretrain:
            self.steps += pretrain

        # Current time step
        self.t = 0

    def __iter__(self):
        return self

    def __next__(self) -> float:
        self.t += 1

        if self.t <= self.pretrain:
            value = self.constant
        else:
            value = max(
                (self.final + (self.init - self.final)
                 * (1 - self.t / (self.steps - self.pretrain))),
                self.final
            )

        # Return sigma
        return value
