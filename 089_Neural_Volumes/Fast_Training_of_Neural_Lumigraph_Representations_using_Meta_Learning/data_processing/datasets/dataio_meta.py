from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


class MetaDatasetSDF(Dataset):
    """
    Meta-dataset for both PCD and IMG view data.
    """

    def __init__(self, datasets: list, opt):
        super().__init__()

        self.opt = opt
        self.SDF_datasets = datasets
        self.dataset_lengths = [len(dataset) for dataset in self.SDF_datasets]

    def __len__(self):
        return sum(self.dataset_lengths)

    def get_dataset_num_and_idx(self, idx):
        dataset_num = 0
        cumulative_length = np.cumsum(self.dataset_lengths)
        for c in range(1, len(cumulative_length)):
            if cumulative_length[c-1] <= idx < cumulative_length[c]:
                dataset_num = c

        if dataset_num == 0:
            relative_idx = idx
        else:
            relative_idx = idx - cumulative_length[dataset_num-1]

        return dataset_num, relative_idx

    def get_SDFDataset(self, dataset_num):
        return self.SDF_datasets[dataset_num]

    def __getitem__(self, idx):
        # get which dataset the idx is in:
        dataset_num, relative_idx = self.get_dataset_num_and_idx(idx)
        DatasetSDF = self.SDF_datasets[dataset_num]

        # Sample PCD and image rays randomly, for the context steps in meta learning. These
        # come from the same dataset as the sampled idx.
        inputs = []
        gt = []
        for _ in range(self.opt.num_meta_steps):
            # idx_rand = np.random.randint(0, len(DatasetSDF))
            idx_rand = np.random.choice(self.opt.TRAIN_VIEWS.copy(), 1).tolist()[0]
            inputs_i, gt_i = DatasetSDF[idx_rand]
            inputs.append(inputs_i)
            gt.append(gt_i)

        # Sample random PCD and image rays to use as meta-supervision, given by idx.
        relative_idx = np.random.choice(self.opt.TRAIN_VIEWS.copy(), 1).tolist()[0]
        inputs_query, gt_query = DatasetSDF[relative_idx]

        meta_batch = {
            'context': {
                'inputs': inputs,
                'gt': gt
            },
            'query': {
                'inputs': inputs_query,
                'gt': gt_query
            },
            'dataset_number': torch.Tensor([dataset_num])
        }

        return meta_batch
