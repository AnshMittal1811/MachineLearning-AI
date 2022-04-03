
import unittest

import torch

import gqnlib


class TestSlimGenerator(unittest.TestCase):

    def test_forward(self):
        model = gqnlib.SlimGenerator()

        x_q = torch.randn(10, 3, 64, 64)
        v_q = torch.randn(10, 4)
        r_c = torch.randn(10, 256, 1, 1)
        canvas, kl_loss = model(x_q, v_q, r_c)

        self.assertTupleEqual(canvas.size(), (10, 3, 64, 64))
        self.assertTupleEqual(kl_loss.size(), (10,))
        self.assertGreater(kl_loss.mean(), 0)

    def test_sample(self):
        model = gqnlib.SlimGenerator()

        v_q = torch.randn(10, 4)
        r_c = torch.randn(10, 256, 1, 1)
        canvas = model.sample(v_q, r_c)

        self.assertTupleEqual(canvas.size(), (10, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
