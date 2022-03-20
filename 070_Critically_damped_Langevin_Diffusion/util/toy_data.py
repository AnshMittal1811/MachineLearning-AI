# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import numpy as np
from sklearn.datasets import make_swiss_roll


def inf_data_gen(dataset, batch_size):
    if dataset == 'multimodal_swissroll':
        NOISE = 0.2
        MULTIPLIER = 0.01
        OFFSETS = [[0.8, 0.8], [0.8, -0.8], [-0.8, -0.8], [-0.8, 0.8]]

        idx = np.random.multinomial(batch_size, [0.2] * 5, size=1)[0]

        sr = []
        for k in range(5):
            sr.append(make_swiss_roll(int(idx[k]), noise=NOISE)[
                      0][:, [0, 2]].astype('float32') * MULTIPLIER)

            if k > 0:
                sr[k] += np.array(OFFSETS[k - 1]).reshape(-1, 2)

        data = np.concatenate(sr, axis=0)[np.random.permutation(batch_size)]
        return torch.from_numpy(data.astype('float32'))

    elif dataset == 'diamond':
        WIDTH = 3
        BOUND = 0.5
        NOISE = 0.04
        ROTATION_MATRIX = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2.)

        means = np.array([(x, y) for x in np.linspace(-BOUND, BOUND, WIDTH)
                          for y in np.linspace(-BOUND, BOUND, WIDTH)])
        means = means @ ROTATION_MATRIX
        covariance_factor = NOISE * np.eye(2)

        index = np.random.choice(
            range(WIDTH ** 2), size=batch_size, replace=True)
        noise = np.random.randn(batch_size, 2)
        data = means[index] + noise @ covariance_factor
        return torch.from_numpy(data.astype('float32'))

    else:
        raise NotImplementedError(
            'Toy dataset %s is not implemented.' % dataset)
