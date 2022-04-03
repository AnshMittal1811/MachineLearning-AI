
import unittest

import torch

import gqnlib


class TestConvolutionalDRAW(unittest.TestCase):

    def setUp(self):
        self.model = gqnlib.ConvolutionalDRAW()

    def test_forward(self):
        x = torch.randn(10, 3, 64, 64)
        v = torch.randn(10, 7)
        r = torch.randn(10, 256, 1, 1)
        canvas, kl_loss = self.model(x, v, r)

        self.assertTupleEqual(canvas.size(), (10, 3, 64, 64))
        self.assertTupleEqual(kl_loss.size(), (10,))
        self.assertGreater(kl_loss.mean(), 0)

    def test_sample(self):
        v = torch.randn(10, 7)
        r = torch.randn(10, 256, 1, 1)
        canvas = self.model.sample(v, r)

        self.assertTupleEqual(canvas.size(), (10, 3, 64, 64))

    def test_forward_alt_r(self):
        x = torch.randn(10, 3, 64, 64)
        v = torch.randn(10, 7)
        r = torch.randn(10, 256, 16, 16)
        canvas, kl_loss = self.model(x, v, r)

        self.assertTupleEqual(canvas.size(), (10, 3, 64, 64))
        self.assertTupleEqual(kl_loss.size(), (10,))
        self.assertGreater(kl_loss.mean(), 0)


if __name__ == "__main__":
    unittest.main()
