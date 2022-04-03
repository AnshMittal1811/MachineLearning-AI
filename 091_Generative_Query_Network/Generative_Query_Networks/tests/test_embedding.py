
import unittest

import torch

import gqnlib


class TestEmbeddingEncoder(unittest.TestCase):

    def test_forward(self):
        vocab_dim = 10
        embed_dim = 8
        n_head = 2
        h_dim = 20
        n_layer = 2
        model = gqnlib.EmbeddingEncoder(
            vocab_dim, embed_dim, n_head, h_dim, n_layer)

        x = torch.arange(vocab_dim).repeat(2).unsqueeze(0)
        x = x.repeat(9, 1)
        batch, length = x.size()
        d = model(x)

        self.assertTupleEqual(d.size(), (batch, embed_dim))


class TestRepresentationNetwork(unittest.TestCase):

    def test_forward(self):
        vocab_dim = 10
        embed_dim = 64
        v_dim = 4
        r_dim = 256
        model = gqnlib.RepresentationNetwork(
            vocab_dim, embed_dim, v_dim, r_dim)

        batch = 9
        c = torch.arange(vocab_dim).repeat(2).unsqueeze(0)
        c = c.repeat(batch, 1)
        v = torch.randn(batch, v_dim)
        d = model(c, v)

        self.assertTupleEqual(d.size(), (batch, r_dim, 1, 1))


if __name__ == "__main__":
    unittest.main()
