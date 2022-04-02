
import unittest

import torch

import gqnlib


class TestUtilMethods(unittest.TestCase):

    def test_nll_normal(self):
        batch = 10
        x_dim = 5

        mu = torch.randn(batch, x_dim)
        var = torch.rand(batch, x_dim) + 0.01
        x = torch.randn(batch, x_dim)

        nll = gqnlib.nll_normal(x, mu, var)
        self.assertTupleEqual(nll.size(), (batch,))
        self.assertTrue((nll >= 0).all())

    def test_kl_batch(self):
        batch = 10
        x_dim = 5

        mu0 = torch.randn(batch, x_dim)
        var0 = torch.rand(batch, x_dim) + 0.01
        mu1 = torch.randn(batch, x_dim)
        var1 = torch.rand(batch, x_dim) + 0.01

        kl = gqnlib.kl_divergence_normal(mu0, var0, mu1, var1)
        self.assertTupleEqual(kl.size(), (batch,))
        self.assertTrue((kl >= 0).all())

    def test_kl_batch_num(self):
        batch = 10
        num_points = 8
        x_dim = 5

        mu0 = torch.randn(batch, num_points, x_dim)
        var0 = torch.rand(batch, num_points, x_dim) + 0.01
        mu1 = torch.randn(batch, num_points, x_dim)
        var1 = torch.rand(batch, num_points, x_dim) + 0.01

        kl = gqnlib.kl_divergence_normal(mu0, var0, mu1, var1)
        self.assertTupleEqual(kl.size(), (batch, num_points))
        self.assertTrue((kl >= 0).all())

    def test_kl_same(self):
        batch = 10
        x_dim = 5

        mu0 = torch.randn(batch, x_dim)
        var0 = torch.rand(batch, x_dim) + 0.01

        kl = gqnlib.kl_divergence_normal(mu0, var0, mu0, var0)
        self.assertTupleEqual(kl.size(), (batch,))
        self.assertTrue((kl == 0).all())


if __name__ == "__main__":
    unittest.main()
