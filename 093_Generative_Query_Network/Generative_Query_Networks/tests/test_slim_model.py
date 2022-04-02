
import unittest

import torch

import gqnlib


class TestSlimGQN(unittest.TestCase):

    def setUp(self):
        self.model = gqnlib.SlimGQN()

    def test_forwad(self):
        d_c = torch.randint(0, 80, (4, 15, 20))
        v_c = torch.randn(4, 15, 4)
        x_q = torch.randn(4, 2, 3, 64, 64)
        v_q = torch.randn(4, 2, 4)

        (canvas, r_c), loss_dict = self.model.inference(
            d_c, v_c, x_q, v_q)

        self.assertTupleEqual(canvas.size(), (4, 2, 3, 64, 64))
        self.assertTupleEqual(r_c.size(), (4, 2, 256, 1, 1))

        self.assertTupleEqual(loss_dict["loss"].size(), (4, 2))
        self.assertTupleEqual(loss_dict["nll_loss"].size(), (4, 2))
        self.assertTupleEqual(loss_dict["kl_loss"].size(), (4, 2))
        self.assertGreater(loss_dict["loss"].mean(), 0)
        self.assertGreater(loss_dict["nll_loss"].mean(), 0)
        self.assertGreater(loss_dict["kl_loss"].mean(), 0)

    def test_forward(self):
        d_c = torch.randint(0, 80, (4, 15, 20))
        v_c = torch.randn(4, 15, 4)
        x_q = torch.randn(4, 2, 3, 64, 64)
        v_q = torch.randn(4, 2, 4)

        loss_dict = self.model(d_c, v_c, x_q, v_q)

        self.assertTupleEqual(loss_dict["loss"].size(), (4, 2))
        self.assertTupleEqual(loss_dict["nll_loss"].size(), (4, 2))
        self.assertTupleEqual(loss_dict["kl_loss"].size(), (4, 2))
        self.assertGreater(loss_dict["loss"].mean(), 0)
        self.assertGreater(loss_dict["nll_loss"].mean(), 0)
        self.assertGreater(loss_dict["kl_loss"].mean(), 0)

    def test_loss_func(self):
        d_c = torch.randint(0, 80, (4, 15, 20))
        v_c = torch.randn(4, 15, 4)
        x_q = torch.randn(4, 2, 3, 64, 64)
        v_q = torch.randn(4, 2, 4)

        loss_dict = self.model.loss_func(d_c, v_c, x_q, v_q)
        self.assertGreater(loss_dict["loss"], 0)
        self.assertGreater(loss_dict["nll_loss"], 0)
        self.assertGreater(loss_dict["kl_loss"], 0)

    def test_reconstruct(self):
        d_c = torch.randint(0, 80, (4, 15, 20))
        v_c = torch.randn(4, 15, 4)
        x_q = torch.randn(4, 1, 3, 64, 64)
        v_q = torch.randn(4, 1, 4)

        canvas = self.model.reconstruct(d_c, v_c, x_q, v_q)
        self.assertTupleEqual(canvas.size(), (4, 1, 3, 64, 64))

    def test_sample(self):
        d_c = torch.randint(0, 80, (4, 15, 20))
        v_c = torch.randn(4, 15, 4)
        v_q = torch.randn(4, 2, 4)

        canvas = self.model.sample(d_c, v_c, v_q)
        self.assertTupleEqual(canvas.size(), (4, 2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
