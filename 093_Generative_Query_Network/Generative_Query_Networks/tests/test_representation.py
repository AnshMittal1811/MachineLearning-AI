
import unittest

import torch

import gqnlib


class TestPyramid(unittest.TestCase):

    def test_pyramid(self):
        batch_size = 10
        x = torch.empty(batch_size, 3, 64, 64)
        v = torch.empty(batch_size, 7)

        model = gqnlib.Pyramid()
        r = model(x, v)

        self.assertTupleEqual(r.size(), (batch_size, 256, 1, 1))


class TestTower(unittest.TestCase):

    def test_tower(self):
        batch_size = 10
        x = torch.empty(batch_size, 3, 64, 64)
        v = torch.empty(batch_size, 7)

        model = gqnlib.Tower(do_pool=False)
        r = model(x, v)

        self.assertTupleEqual(r.size(), (batch_size, 256, 16, 16))

    def test_pool(self):
        batch_size = 10
        x = torch.empty(batch_size, 3, 64, 64)
        v = torch.empty(batch_size, 7)

        model = gqnlib.Tower(do_pool=True)
        r = model(x, v)

        self.assertTupleEqual(r.size(), (batch_size, 256, 1, 1))


class TestSimple(unittest.TestCase):

    def test_simple(self):
        batch_size = 10
        x = torch.empty(batch_size, 3, 64, 64)
        v = torch.empty(batch_size, 7)

        model = gqnlib.Simple()
        r = model(x, v)

        self.assertTupleEqual(r.size(), (batch_size, 32, 16, 16))


if __name__ == "__main__":
    unittest.main()
