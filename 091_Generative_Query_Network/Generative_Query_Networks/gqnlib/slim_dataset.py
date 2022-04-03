
"""Dataset class for SLIM (Spatial Language Integrating Model).

ref)
https://github.com/deepmind/slim-dataset
"""

from typing import Tuple, List, DefaultDict

import collections
import gzip
import json
import logging
import pathlib
import random

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class WordVectorizer:
    """Word-vector encoder.

    ref)
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    Args:
        vocab_dim (int, optional): Max dimension of vocabularies.

    Attributes:
        word2index (collections.defaultdict): Word to index dict.
        word2count (collections.defaultdict): Word to counts dict.
        index2word (collections.defaultdict): Index to word dict.
        n_words (int): Number of words.
    """

    def __init__(self, vocab_dim: int = 5000) -> None:

        self.vocab_dim = vocab_dim

        self.word2index: DefaultDict[str, int] = collections.defaultdict(int)
        self.word2count: DefaultDict[str, int] = collections.defaultdict(int)
        self.index2word = {0: "UNK", 1: "SOS", 2: "EOS"}
        self.n_words = 3

        self.removal = """!@#$%^&*()_-+=|¥~`[]{}:;"',.<>/?"""

    def __len__(self) -> int:
        """Length of registered words."""
        return self.n_words

    def sentence2index(self, sentence: str, register: bool = True
                       ) -> List[int]:
        """Convert sentence to indices list.

        **Caution**: if `register` is `False` and unknown word is given, the
        word is registered as a key of `self.word2index` dict, but
        `self.n_words` is not incremented.

        Args:
            sentence (str): Sentence string.
            register (bool): If true, unknown words are registered to dict.

        Returns:
            indices (list of int): Indices list.
        """

        indices = []
        for word in sentence.split(" "):
            # Preprocess
            word = word.lower().strip(self.removal)

            # Ignore empty string
            if not word:
                continue

            # Register
            if register:
                if (word not in self.word2index) and \
                        (self.n_words <= self.vocab_dim):
                    self.word2index[word] = self.n_words
                    self.word2count[word] = 1
                    self.index2word[self.n_words] = word
                    self.n_words += 1
                else:
                    self.word2count[word] += 1

            # Get index
            indices.append(self.word2index[word])

        return indices

    def to_json(self, path: str) -> None:
        """Saves to json file.

        Args:
            path (str): Path to saved file.
        """

        data = {
            "word2index": self.word2index,
            "word2count": self.word2count,
            "index2word": self.index2word,
            "n_words": self.n_words,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def read_json(self, path: str) -> None:
        """Reads saved json file.

        Args:
            path (str): Path to json file.
        """

        with open(path, "r") as f:
            data = json.load(f)

        word2index = {k: int(v) for k, v in data["word2index"].items()}
        word2count = {k: int(v) for k, v in data["word2count"].items()}

        self.word2index = collections.defaultdict(int, word2index)
        self.word2count = collections.defaultdict(int, word2count)
        self.index2word = {int(k): v for k, v in data["index2word"].items()}
        self.n_words = int(data["n_words"])

    def read_ptgz(self, path: str) -> None:
        """Reads '*.pt.gz' file.

        '*.pt.gz' file is expexted to include a list of tuples of tensors.

        Args:
            path (str): Path to 'pt.gz' file.
        """

        with gzip.open(path, "rb") as f:
            dataset = torch.load(f)

        for _, _, _, cpt, *_ in dataset:
            for snt in cpt:
                self.sentence2index(snt.decode())


class SlimDataset(torch.utils.data.Dataset):
    """SlimDataset class for SLIM.

    SlimDataset class loads data files at each time accessed by index. Each
    `*.pt.gz` file includes list of tuples, and these are rearanged to mini
    batches.

    Args:
        root_dir (str): Path to root directory.
        batch_size (int): Batch size.
        vectorizer (WordVectorizer): Pre-trained vectorizer.
        train (bool, optional): If `True`, register read sentences to
            vectorizer.

    Attributes:
        record_list (list of pathlib.Path): List of path to data files.
    """

    def __init__(self, root_dir: str, batch_size: int,
                 vectorizer: WordVectorizer, train: bool = True) -> None:
        super().__init__()

        self.record_list = sorted(pathlib.Path(root_dir).glob("*.pt.gz"))
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        self.train = train

        self.logger = logging.getLogger()

        if not self.record_list:
            self.logger.warning("No record exists in root directory.")

    def __len__(self) -> int:
        """Returns number of files and directories in root dir.

        Returns:
            len (int): Number of objects in root dir.
        """

        return len(self.record_list)

    def __getitem__(self, index: int) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """Loads data file and returns data with specified index.

        This method reads `<index>.pt.gz` file which includes a list of tuples
        `(images, viewpoints, topdown, captions, *)`, and returns list of
        tuples of tensors `(images, viewpoints, captions)`.

        * Image size: `(b, m, 3, 64, 64)`
        * Viewpoints size: `(b, m, 4)`
        * Captions size: `(b, m, l)`

        Args:
            index (int): Index number.

        Returns:
            data_list (torch.Tensor): List of tuples of tensors
                `(images, viewpoints, captions)`. Length of list is
                `data_num // batch_size`.
        """

        try:
            with gzip.open(self.record_list[index], "rb") as f:
                dataset = torch.load(f)
        except (UnicodeDecodeError, ValueError) as e:
            self.logger.debug(f"Invalid file {self.record_list[index]}: {e}")
            return []

        # Read list of tuples
        images_list = []
        viewpoints_list = []
        captions_list = []
        for img, vwp, _, cpt, *_ in dataset:
            images_list.append(torch.from_numpy(img).permute(0, 3, 1, 2))
            viewpoints_list.append(torch.from_numpy(vwp))

            sentences = []
            for snt in cpt:
                sentences.append(torch.tensor(
                    self.vectorizer.sentence2index(snt.decode(), self.train)))

            captions_list.append(pad_sequence(sentences, batch_first=False))

        # Stack loaded tensors (n, m, *)
        images = torch.stack(images_list)
        viewpoints = torch.stack(viewpoints_list)
        captions = pad_sequence(captions_list).permute(1, 2, 0)

        # Trim off extra elements
        batch_num = images.size(0) // self.batch_size
        images = images[:self.batch_size * batch_num]
        viewpoints = viewpoints[:self.batch_size * batch_num]
        captions = captions[:self.batch_size * batch_num]

        _, *i_dims = images.size()
        _, *v_dims = viewpoints.size()
        _, *c_dims = captions.size()

        # Resize: (n, m, *) -> (a, b, m, *)
        images = images.contiguous().view(
            batch_num, self.batch_size, *i_dims)
        viewpoints = viewpoints.contiguous().view(
            batch_num, self.batch_size, *v_dims)
        captions = captions.contiguous().view(
            batch_num, self.batch_size, *c_dims)

        data_list = []
        for i in range(batch_num):
            data_list.append((images[i], viewpoints[i], captions[i]))

        return data_list


def partition_slim(images: Tensor, viewpoints: Tensor, captions: Tensor,
                   num_query: int = 1, num_context: int = -1
                   ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Partitions given SLIM data in context and query sets.

    * Context: (captions_context, viewpoints_context)
    * Query: (images_query, viewpoints_query)

    Args:
        images (torch.Tensor): Image tensor, size
            `(n, b, num_points, c, h, w)`.
        viewpoints (torch.Tensor): Viewpoints tensor, size
            `(n, b, num_points, target)`.
        captions (torch.Tensor): Captions tensor, size
            `(n, b, num_points, length)`.
        num_query (int, optional): Number of queries.
        num_context (int, optional): Number of contexts. If the value is less
            than 1, random number is sampled.

    Returns:
        d_c (torch.Tensor): Context captions, size `(n*b, num_context, l)`.
        v_c (torch.Tensor): Context viewpoints, size `(n*b, num_context, t)`.
        x_q (torch.Tensor): Query images, size `(n*b, num_query, c, h, w)`.
        v_q (torch.Tensor): Query viewpoints, size `(n*b, num_query, t)`.

    Raises:
        ValueError: If given image dimension is not 5 or 6.
        ValueError: If `num_query` is equal or greater than `num_points`.
    """

    # Data size
    if images.dim() == 6:
        n, b, num_points, *x_dims = images.size()
        _, _, _, *v_dims = viewpoints.size()
        _, _, _, *d_dims = captions.size()
    elif images.dim() == 5:
        n = 1
        b, num_points, *x_dims = images.size()
        _, _, *v_dims = viewpoints.size()
        _, _, *d_dims = captions.size()
    else:
        raise ValueError("Given image size is expected to be (n, b, m, c, h, "
                         f"w) or (b, m, c, h, w), but given {images.size()}")

    if num_query >= num_points:
        raise ValueError(f"Number of queries (n={num_query}) must be less "
                         f"than total data (n={num_points}).")

    # Squeeze dataset
    images = images.view(n*b, num_points, *x_dims)
    viewpoints = viewpoints.view(n*b, num_points, *v_dims)
    captions = captions.view(n*b, num_points, *d_dims)

    # Sample randum number for total data size
    if num_context > 0:
        n_data = min(num_context + num_query, num_points)
    else:
        n_data = random.randint(num_query + 1, num_points)

    # Shuffle indices
    indices = random.sample(range(num_points), n_data)

    # Partition into context and query
    context_idx = indices[:-num_query]
    query_idx = indices[-num_query:]

    d_c = captions[:, context_idx]
    v_c = viewpoints[:, context_idx]

    x_q = images[:, query_idx]
    v_q = viewpoints[:, query_idx]

    return d_c, v_c, x_q, v_q
