
import unittest

import torch

import gqnlib


class TestDictionaryEncoder(unittest.TestCase):

    def test_foward(self):

        model = gqnlib.DictionaryEncoder()

        x = torch.randn(10, 3, 64, 64)
        v = torch.randn(10, 7)
        key, value = model(x, v)

        self.assertTupleEqual(key.size(), (490, 64, 8, 8))
        self.assertTupleEqual(value.size(), (490, 76, 8, 8))


class TestAttentionGenerator(unittest.TestCase):

    def test_forward(self):
        model = gqnlib.AttentionGenerator()

        x = torch.randn(9, 3, 64, 64)
        v = torch.randn(9, 7)
        key = torch.randn(490, 64, 8, 8)
        value = torch.randn(490, 76, 8, 8)

        canvas, r_stack, kl_loss = model(x, v, key, value)

        self.assertTupleEqual(canvas.size(), (9, 3, 64, 64))
        self.assertTupleEqual(r_stack.size(), (9, 76, 8, 8))
        self.assertTupleEqual(kl_loss.size(), (9,))
        self.assertTrue((kl_loss > 0).all())

    def test_sample(self):
        model = gqnlib.AttentionGenerator()

        v = torch.randn(9, 7)
        key = torch.randn(490, 64, 8, 8)
        value = torch.randn(490, 76, 8, 8)

        canvas, r_stack = model.sample(v, key, value)
        self.assertTupleEqual(canvas.size(), (9, 3, 64, 64))
        self.assertTupleEqual(r_stack.size(), (9, 76, 8, 8))


if __name__ == "__main__":
    unittest.main()
