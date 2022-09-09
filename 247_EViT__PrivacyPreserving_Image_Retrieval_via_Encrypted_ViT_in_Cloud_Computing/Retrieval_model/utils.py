import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random
import copy


def split_data(type='Corel10-a'):
    Coefficient_bit_features = np.load('data/difffeature_matrix.npy', allow_pickle=True).astype(np.float16)
    huffman_code = np.load('data/huffman_feature.npy', allow_pickle=True).astype(np.float32)
    label = []
    for i in range(100):
        for j in range(100):
            label.append(i)
    if type == 'Corel10-a':
        train_data = Coefficient_bit_features[:7000, :, :]
        train_label = label[:7000]

        test_data = Coefficient_bit_features[7000:, :, :]
        test_label = label[7000:]

        train_huffman_feature = huffman_code[:7000, :]
        test_huffman_feature = huffman_code[7000:, :]
    else:
        train_data, test_data, train_label, test_label = train_test_split(Coefficient_bit_features, label, test_size=0.3, stratify=label, random_state=2020)
        train_huffman_feature, test_huffman_feature, train_label, test_label = train_test_split(huffman_code, label, test_size=0.3, stratify=label, random_state=2020)
    return train_data, test_data, train_huffman_feature, test_huffman_feature, train_label, test_label


# PK采样
class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def BaseDataset(tr_path, tr_label):
    train_sample_data = []
    for path, label in zip(tr_path, tr_label):
        train_sample_data.append((path, label))

    return train_sample_data
