# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from . import utils


@utils.register_model(name='mlp')
class MLP(nn.Module):
    def __init__(self,
                 config,
                 input_dim=2,
                 index_dim=1,
                 hidden_dim=128):

        super().__init__()

        act = nn.SiLU()

        self.x_input = True
        self.v_input = True if config.sde == 'cld' else False

        if self.x_input and self.v_input:
            in_dim = input_dim * 2 + index_dim
        else:
            in_dim = input_dim + index_dim
        out_dim = input_dim

        self.main = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                  act,
                                  nn.Linear(hidden_dim, hidden_dim),
                                  act,
                                  nn.Linear(hidden_dim, hidden_dim),
                                  act,
                                  nn.Linear(hidden_dim, hidden_dim),
                                  act,
                                  nn.Linear(hidden_dim, out_dim))

    def forward(self, u, t):
        h = torch.cat([u, t.reshape(-1, 1)], dim=1)
        output = self.main(h)

        return output
