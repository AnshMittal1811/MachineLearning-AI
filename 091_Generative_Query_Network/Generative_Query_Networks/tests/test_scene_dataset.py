
import unittest

import gzip
import pathlib
import tempfile

import torch

import gqnlib


class TestSceneDataset(unittest.TestCase):

    def test_len(self):
        dataset = gqnlib.SceneDataset(".", 10)
        self.assertGreaterEqual(len(dataset), 0)

    def test_getitem(self):
        # Dummy data
        imgs = torch.empty(4, 64, 64, 3)
        tgts = torch.empty(4, 5)
        data = [(imgs.numpy(), tgts.numpy())] * 10

        with tempfile.TemporaryDirectory() as root:
            path = str(pathlib.Path(root, "1.pt.gz"))
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            # Access data
            dataset = gqnlib.SceneDataset(root, 5)
            data_list = dataset[0]

        self.assertEqual(len(data_list), 2)

        frames, viewpoints = data_list[0]
        self.assertTupleEqual(frames.size(), (5, 4, 3, 64, 64))
        self.assertTupleEqual(viewpoints.size(), (5, 4, 7))

    def test_multiitem(self):
        # Dummy data
        imgs = torch.empty(4, 64, 64, 3)
        tgts = torch.empty(4, 5)
        data = [(imgs.numpy(), tgts.numpy())] * 10

        with tempfile.TemporaryDirectory() as root:
            path = str(pathlib.Path(root, "1.pt.gz"))
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            path = str(pathlib.Path(root, "2.pt.gz"))
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            # Access data
            dataset = gqnlib.SceneDataset(root, 5)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            batch = next(iter(loader))

        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], list)
        self.assertIsInstance(batch[0][0], torch.Tensor)
        self.assertTupleEqual(batch[0][0].size(), (2, 5, 4, 3, 64, 64))

    def test_partition_scene(self):
        # Data
        images = torch.empty(1, 5, 15, 3, 64, 64)
        viewpoints = torch.empty(1, 5, 15, 7)

        # Query single data
        x_c, v_c, x_q, v_q = gqnlib.partition_scene(images, viewpoints)

        # x_c
        self.assertEqual(x_c.size(0), 5)
        self.assertTrue(0 < x_c.size(1) < 15)
        self.assertEqual(x_c.size(2), 3)
        self.assertEqual(x_c.size(3), 64)
        self.assertEqual(x_c.size(4), 64)

        # v_c
        self.assertEqual(v_c.size(0), 5)
        self.assertTrue(0 < x_c.size(1) < 15)
        self.assertEqual(v_c.size(2), 7)

        # Query
        self.assertTupleEqual(x_q.size(), (5, 1, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, 1, 7))

        # Query multiple data
        num_query = 14
        x_c, v_c, x_q, v_q = gqnlib.partition_scene(
            images, viewpoints, num_query=num_query)

        # x_c
        self.assertEqual(x_c.size(0), 5)
        self.assertTrue(0 < x_c.size(1) < 15)
        self.assertEqual(x_c.size(2), 3)
        self.assertEqual(x_c.size(3), 64)
        self.assertEqual(x_c.size(4), 64)

        # v_c
        self.assertEqual(v_c.size(0), 5)
        self.assertTrue(0 < v_c.size(1) < 15)
        self.assertEqual(v_c.size(2), 7)

        self.assertTupleEqual(x_q.size(), (5, num_query, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, num_query, 7))

        # Query size is too largs
        with self.assertRaises(ValueError):
            gqnlib.partition_scene(images, viewpoints, num_query=15)

    def test_partition_scene_multi(self):
        # Data
        images = torch.empty(2, 5, 15, 3, 64, 64)
        viewpoints = torch.empty(2, 5, 15, 7)

        # Query single data
        x_c, v_c, x_q, v_q = gqnlib.partition_scene(images, viewpoints)

        # Context
        self.assertEqual(x_c.size(0), 10)
        self.assertTrue(0 < x_c.size(1) < 15)
        self.assertEqual(v_c.size(0), 10)

        # Query
        self.assertTupleEqual(x_q.size(), (10, 1, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (10, 1, 7))

    def test_partition_scene_single(self):
        # Data
        images = torch.empty(5, 15, 3, 64, 64)
        viewpoints = torch.empty(5, 15, 7)

        # Query single data
        x_c, v_c, x_q, v_q = gqnlib.partition_scene(images, viewpoints)

        # Context
        self.assertEqual(x_c.size(0), 5)
        self.assertTrue(0 < x_c.size(1) < 15)
        self.assertEqual(v_c.size(0), 5)

        # Query
        self.assertTupleEqual(x_q.size(), (5, 1, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, 1, 7))

    def test_partition_scene_fixed(self):
        # Data
        images = torch.empty(5, 15, 3, 64, 64)
        viewpoints = torch.empty(5, 15, 7)

        # Query single data
        x_c, v_c, x_q, v_q = gqnlib.partition_scene(
            images, viewpoints, num_query=2, num_context=4)

        # Context
        self.assertTupleEqual(x_c.size(), (5, 4, 3, 64, 64))
        self.assertTupleEqual(v_c.size(), (5, 4, 7))

        # Query
        self.assertTupleEqual(x_q.size(), (5, 2, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, 2, 7))

        # Oversized number

        # Query single data
        x_c, v_c, x_q, v_q = gqnlib.partition_scene(
            images, viewpoints, num_query=2, num_context=200)

        # Context
        self.assertTupleEqual(x_c.size(), (5, 13, 3, 64, 64))
        self.assertTupleEqual(v_c.size(), (5, 13, 7))

        # Query
        self.assertTupleEqual(x_q.size(), (5, 2, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, 2, 7))


if __name__ == "__main__":
    unittest.main()
