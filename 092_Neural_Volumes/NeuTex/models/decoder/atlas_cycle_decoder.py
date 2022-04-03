import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..networks import init_weights, init_seq


class Mapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    Modified : Fanbo Xiang 2020
    """

    def __init__(self, code_size, input_point_dim, hidden_size=128, num_layers=2):
        """
        template_size: input size
        """
        super().__init__()
        self.code_size = code_size
        self.input_size = input_point_dim
        self.dim_output = 3
        self.hidden_neurons = hidden_size
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.input_size, self.code_size)
        self.linear2 = nn.Linear(self.code_size, self.hidden_neurons)

        init_weights(self.linear1)
        init_weights(self.linear2)

        self.linear_list = nn.ModuleList(
            [
                nn.Linear(self.hidden_neurons, self.hidden_neurons)
                for i in range(self.num_layers)
            ]
        )

        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(self.hidden_neurons, self.dim_output)
        init_weights(self.last_linear)

        self.activation = F.relu

    def forward(self, x, latent):
        x = self.linear1(x) + latent
        x = self.activation(x)
        x = self.activation(self.linear2(x))
        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        return self.last_linear(x)


class AtlasCycleDecoder(nn.Module):
    def __init__(self, num_primitives, code_size):
        super().__init__()

        self.num_primitives = num_primitives

        # Intialize deformation networks
        self.decoder = nn.ModuleList(
            [Mapping2Dto3D(code_size, 2) for i in range(0, num_primitives)]
        )

    def forward(self, uv, code):
        points = [
            self.decoder[i](uv[..., i, :], code[:, None, None, :])
            for i in range(0, self.num_primitives)
        ]

        return torch.stack(points, dim=-2)
