# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License-NC
# See LICENSE.txt for details
#
# Author: Zheng Tang (tangzhengthomas@gmail.com)


from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_vids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, vid, _, _, _, _, _, _) in enumerate(self.data_source):
            self.index_dic[vid].append(index)
        self.vids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for vid in self.vids:
            idxs = self.index_dic[vid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for vid in self.vids:
            idxs = copy.deepcopy(self.index_dic[vid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[vid].append(batch_idxs)
                    batch_idxs = []

        avai_vids = copy.deepcopy(self.vids)
        final_idxs = []

        while len(avai_vids) >= self.num_vids_per_batch:
            selected_vids = random.sample(avai_vids, self.num_vids_per_batch)
            for vid in selected_vids:
                batch_idxs = batch_idxs_dict[vid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[vid]) == 0:
                    avai_vids.remove(vid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
