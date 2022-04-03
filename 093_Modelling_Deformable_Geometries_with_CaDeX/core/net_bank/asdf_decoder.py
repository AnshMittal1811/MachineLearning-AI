#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# from A-SDF implementation
# add occ option

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size=253,
        dims=[512, 512, 512, 768, 512, 512, 512, 512],
        dropout=[0, 1, 2, 3, 4, 5, 6, 7],
        dropout_prob=0.2,
        norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
        latent_in=[4],
        articulation=True,  # for development
        num_atc_parts=1,
        do_sup_with_part=False,
        occ_flag=True,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        if articulation == True:
            dims = [latent_size + 3] + dims + [1]
            # original code has flaw, this should not be fixed as 256, should be dims[0]
            self.fc1 = nn.utils.weight_norm(nn.Linear(num_atc_parts + 3, dims[0]))
        else:
            dims = [latent_size + 3] + dims + [1]

        if do_sup_with_part:
            self.part_fc = nn.utils.weight_norm(nn.Linear(512, num_atc_parts + 2))

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in

        self.num_atc_parts = num_atc_parts
        self.do_sup_with_part = do_sup_with_part

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0] * 2
            else:
                out_dim = dims[layer + 1]

            if layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        if not occ_flag:
            self.th = nn.Tanh()
        print(self)

    # input: N x (L+3+num_atc_parts)
    def forward(self, input):

        xyz_atc = input[:, -(self.num_atc_parts + 3) :]  # xyz + articulation (3+1)
        # xyz_atc[:, 3:] = xyz_atc[:, 3:] / 100
        xyz_atc = self.fc1(xyz_atc)
        atc_emb = self.relu(xyz_atc)

        xyz_shape = input[:, : -self.num_atc_parts]
        x = xyz_shape

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, xyz_shape, atc_emb], 1)

            if layer < self.num_layers - 2:
                x = lin(x)

            if layer < self.num_layers - 2:
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.do_sup_with_part:
            pred_part = self.part_fc(x)
            x = lin(x)
        else:
            x = lin(x)

        if hasattr(self, "th"):
            x = self.th(x)

        if self.do_sup_with_part:
            return x, pred_part
        else:
            return x
