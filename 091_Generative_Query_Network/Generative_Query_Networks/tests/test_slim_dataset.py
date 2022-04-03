
import unittest

import gzip
import json
import pathlib
import tempfile

import torch

import gqnlib


class TestWordVectorizer(unittest.TestCase):

    def setUp(self):
        self.vectorizer = gqnlib.WordVectorizer()

    def test_sentence2index(self):
        sentence = "aA, ab. aa? aa! ba,.,., ba!?"
        indices = self.vectorizer.sentence2index(sentence)

        # Indices
        self.assertListEqual(indices, [3, 4, 3, 3, 5, 5])

        # Word to index
        self.assertSetEqual(set(self.vectorizer.word2index.keys()),
                            set(["aa", "ab", "ba"]))

        self.assertEqual(self.vectorizer.word2index["aa"], 3)
        self.assertEqual(self.vectorizer.word2index["ab"], 4)
        self.assertEqual(self.vectorizer.word2index["ba"], 5)

        self.assertEqual(self.vectorizer.word2index["c"], 0)
        self.assertEqual(self.vectorizer.word2index["d"], 0)

        # Word to count
        self.assertEqual(self.vectorizer.word2count["aa"], 3)
        self.assertEqual(self.vectorizer.word2count["ab"], 1)
        self.assertEqual(self.vectorizer.word2count["ba"], 2)

        self.assertEqual(self.vectorizer.word2count["c"], 0)
        self.assertEqual(self.vectorizer.word2count["d"], 0)

        # Index to word
        self.assertEqual(self.vectorizer.index2word[0], "UNK")
        self.assertEqual(self.vectorizer.index2word[3], "aa")
        self.assertEqual(self.vectorizer.index2word[4], "ab")

        # N words
        self.assertEqual(self.vectorizer.n_words, 6)

        # Length
        self.assertEqual(len(self.vectorizer), 6)

    def test_max_vocab(self):
        self.vectorizer.vocab_dim = 4

        sentence = "aA, ab. aa? aa! ba,.,., ba!?"
        _ = self.vectorizer.sentence2index(sentence)

        self.assertSetEqual(set(self.vectorizer.word2index.keys()),
                            set(["aa", "ab", "ba"]))

        self.assertEqual(self.vectorizer.word2index["aa"], 3)
        self.assertEqual(self.vectorizer.word2index["ab"], 4)
        self.assertEqual(self.vectorizer.word2index["ba"], 0)

    def test_sentence2index_no_register(self):
        sentence = "aA, ab. aa? aa! ba,.,., ba!?"
        self.vectorizer.sentence2index(sentence)

        sentence = "ac, aa., BA?"
        indices = self.vectorizer.sentence2index(sentence, register=False)

        self.assertListEqual(indices, [0, 3, 5])
        self.assertEqual(self.vectorizer.word2count["aa"], 3)
        self.assertTrue("ac" in self.vectorizer.word2index)
        self.assertTrue("ac" not in self.vectorizer.word2count)

    def test_to_json(self):
        sentence = "aa, ab. aa? aa! ba,.,., ba!?"
        self.vectorizer.sentence2index(sentence)

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.json")
            self.vectorizer.to_json(path)

            with path.open() as f:
                data = json.load(f)

        self.assertTrue("word2index" in data)
        self.assertTrue("word2count" in data)
        self.assertTrue("index2word" in data)
        self.assertTrue("n_words" in data)

        self.assertDictEqual(data["word2index"], {"aa": 3, "ab": 4, "ba": 5})

    def test_read_json(self):
        sentence = "aa, ab. aa? aa! ba,.,., ba!?"
        self.vectorizer.sentence2index(sentence)

        model = gqnlib.WordVectorizer()

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.json")
            self.vectorizer.to_json(path)

            model.read_json(path)

        self.assertDictEqual(self.vectorizer.word2index, model.word2index)
        self.assertDictEqual(self.vectorizer.word2count, model.word2count)
        self.assertDictEqual(self.vectorizer.index2word, model.index2word)
        self.assertEqual(self.vectorizer.n_words, model.n_words)

    def test_read_ptgz(self):
        c = torch.randn(10, 3, 64, 64).numpy()
        cpt = [b"aa, ab. aa? aa! ba,.,., ba!? " * 10]
        data = [(c, c, c, cpt, c, c)] * 10

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.json")
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            self.vectorizer.read_ptgz(path)

        self.assertEqual(self.vectorizer.word2index["aa"], 3)
        self.assertEqual(self.vectorizer.word2count["aa"], 300)
        self.assertEqual(self.vectorizer.index2word[3], "aa")
        self.assertEqual(self.vectorizer.n_words, 6)


class TestSlimDataset(unittest.TestCase):

    def test_len(self):
        batch_size = 10
        dataset = gqnlib.SlimDataset(".", batch_size, None)
        self.assertGreaterEqual(len(dataset), 0)

    def test_getitem(self):
        # Vectorizer
        vectorizer = gqnlib.WordVectorizer()
        vectorizer.sentence2index("aa, ab. aa? aa! ba,.,., ba!?")

        # Dataset
        imgs = torch.empty(8, 64, 64, 3).numpy()
        tgts = torch.empty(8, 4).numpy()
        cpts = [b"aa, ab. aa? aa! ba,.,., ba!?"] * 8
        data = [(imgs, tgts, tgts, cpts, tgts)] * 20

        # Save and load
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.pt.gz")
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            # Get data item
            dataset = gqnlib.SlimDataset(root, 10, vectorizer)
            data_list = dataset[0]

        self.assertEqual(len(data_list), 2)

        frames, viewpoints, captions = data_list[0]
        self.assertTupleEqual(frames.size(), (10, 8, 3, 64, 64))
        self.assertTupleEqual(viewpoints.size(), (10, 8, 4))
        self.assertTupleEqual(captions.size(), (10, 8, 6))

    def test_multi_getitem(self):
        # Vectorizer
        vectorizer = gqnlib.WordVectorizer()
        vectorizer.sentence2index("aa, ab. aa? aa! ba,.,., ba!?")

        # Dataset
        imgs = torch.empty(8, 64, 64, 3).numpy()
        tgts = torch.empty(8, 4).numpy()
        cpts = [b"aa, ab. aa? aa! ba,.,., ba!?"] * 8
        data = [(imgs, tgts, tgts, cpts, tgts)] * 20

        # Save and load
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.pt.gz")
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            path = pathlib.Path(root, "tmp1.pt.gz")
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            # Get data item
            dataset = gqnlib.SlimDataset(root, 10, vectorizer)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            batch = next(iter(loader))

        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], list)
        self.assertIsInstance(batch[0][0], torch.Tensor)
        self.assertTupleEqual(batch[0][0].size(), (2, 10, 8, 3, 64, 64))
        self.assertTupleEqual(batch[0][1].size(), (2, 10, 8, 4))
        self.assertTupleEqual(batch[0][2].size(), (2, 10, 8, 6))

    def test_partition_slim(self):
        images = torch.empty(1, 5, 15, 3, 64, 64)
        viewpoints = torch.empty(1, 5, 15, 4)
        captions = torch.empty(1, 5, 15, 20)

        # Single query
        d_c, v_c, x_q, v_q = gqnlib.partition_slim(
            images, viewpoints, captions, num_context=100)

        # Context
        self.assertTupleEqual(d_c.size(), (5, 14, 20))
        self.assertTupleEqual(v_c.size(), (5, 14, 4))

        # Query
        self.assertTupleEqual(x_q.size(), (5, 1, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, 1, 4))

        # Randomized
        d_c, v_c, x_q, v_q = gqnlib.partition_slim(
            images, viewpoints, captions)

        # d_c
        self.assertEqual(d_c.size(0), 5)
        self.assertTrue(0 < d_c.size(1) < 15)
        self.assertEqual(d_c.size(2), 20)

        # v_c
        self.assertEqual(v_c.size(0), 5)
        self.assertTrue(0 < d_c.size(1) < 15)
        self.assertEqual(v_c.size(2), 4)

        # Query
        self.assertTupleEqual(x_q.size(), (5, 1, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, 1, 4))

        # Query multiple data
        num_query = 14
        d_c, v_c, x_q, v_q = gqnlib.partition_slim(
            images, viewpoints, captions, num_query=num_query)

        # d_c
        self.assertEqual(d_c.size(0), 5)
        self.assertEqual(d_c.size(1), 1)
        self.assertEqual(d_c.size(2), 20)

        # v_c
        self.assertEqual(v_c.size(0), 5)
        self.assertTrue(0 < v_c.size(1) < 15)
        self.assertEqual(v_c.size(2), 4)

        self.assertTupleEqual(x_q.size(), (5, num_query, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, num_query, 4))

        # Query size is too largs
        with self.assertRaises(ValueError):
            gqnlib.partition_slim(
                images, viewpoints, captions, num_query=15)

    def test_partition_slim_alt(self):
        # Multiple data
        images = torch.empty(2, 5, 15, 3, 64, 64)
        viewpoints = torch.empty(2, 5, 15, 4)
        captions = torch.empty(2, 5, 15, 20)

        # Single query
        d_c, v_c, x_q, v_q = gqnlib.partition_slim(
            images, viewpoints, captions, num_context=14)

        # Size check
        self.assertTupleEqual(d_c.size(), (10, 14, 20))
        self.assertTupleEqual(v_c.size(), (10, 14, 4))
        self.assertTupleEqual(x_q.size(), (10, 1, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (10, 1, 4))

        # n = 1
        images = torch.empty(5, 15, 3, 64, 64)
        viewpoints = torch.empty(5, 15, 4)
        captions = torch.empty(5, 15, 20)

        # Single query
        d_c, v_c, x_q, v_q = gqnlib.partition_slim(
            images, viewpoints, captions, num_context=1000)

        # Size check
        self.assertTupleEqual(d_c.size(), (5, 14, 20))
        self.assertTupleEqual(v_c.size(), (5, 14, 4))
        self.assertTupleEqual(x_q.size(), (5, 1, 3, 64, 64))
        self.assertTupleEqual(v_q.size(), (5, 1, 4))

        # Raise
        with self.assertRaises(ValueError):
            images = torch.empty(15, 3, 64, 64)
            viewpoints = torch.empty(15, 4)
            captions = torch.empty(15, 20)

            _ = gqnlib.partition_slim(images, viewpoints, captions)


if __name__ == "__main__":
    unittest.main()
