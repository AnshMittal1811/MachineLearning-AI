
import unittest

import torch
from torch import nn, optim

import gqnlib


class TestAnnealingStepLR(unittest.TestCase):

    def test_step(self):
        net = DummyNet()
        optimizer = optim.Adam(net.parameters())
        annealr = gqnlib.AnnealingStepLR(optimizer, mu_i=0.1, mu_f=0.01, n=10)
        x = torch.randn(8, 10)

        net.train()
        for i in range(12):
            loss = (net(x) - 1).mean()
            loss.backward()
            optimizer.step()
            annealr.step()

            lr = max(0.01 + (0.1 - 0.01) * (1.0 - (i + 1) / 10), 0.01)
            for group in optimizer.param_groups:
                self.assertEqual(group["lr"], lr)


class TestAnnealer(unittest.TestCase):

    def test_iter(self):

        annealer = gqnlib.Annealer(0.1, 0.01, 10)
        for i in range(12):
            val = next(annealer)
            true = max(0.01 + (0.1 - 0.01) * (1.0 - (i + 1) / 10), 0.01)
            self.assertEqual(val, true)


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    unittest.main()
