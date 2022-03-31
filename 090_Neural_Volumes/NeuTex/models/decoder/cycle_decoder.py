import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..networks import init_seq, init_weights



class CycleDecoder(nn.Module):
    def __init__(self, uv_dim, code_dim):
        super().__init__()
        self.uv_dim = uv_dim
        self.code_dim = code_dim

        if self.code_dim > 0:
            self.in_linear = nn.Linear(self.uv_dim, self.code_dim)
            block = [
                nn.Linear(self.code_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
            ]
            init_weights(self.in_linear)
        else:
            block = [
                nn.Linear(self.uv_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
            ]

        self.block = nn.Sequential(*block)

        rblock1 = []
        for i in range(2):
            rblock1.append(nn.Linear(256, 256))
            rblock1.append(nn.LeakyReLU(0.2))
        rblock1.append(nn.Linear(256, 256))
        self.rblock1 = nn.Sequential(*rblock1)

        rblock2 = []
        for i in range(2):
            rblock2.append(nn.Linear(256, 256))
            rblock2.append(nn.LeakyReLU(0.2))
        rblock2.append(nn.Linear(256, 256))
        self.rblock2 = nn.Sequential(*rblock2)

        rblock3 = []
        for i in range(2):
            rblock3.append(nn.Linear(256, 256))
            rblock3.append(nn.LeakyReLU(0.2))
        rblock3.append(nn.Linear(256, 256))
        self.rblock3 = nn.Sequential(*rblock3)

        rblock4 = []
        for i in range(2):
            rblock4.append(nn.Linear(256, 256))
            rblock4.append(nn.LeakyReLU(0.2))
        rblock4.append(nn.Linear(256, 256))
        self.rblock4 = nn.Sequential(*rblock4)

        self.block2 = nn.Sequential(nn.Linear(256, 3), nn.Tanh())

        for b in [
            self.block,
            self.rblock1,
            self.rblock2,
            self.rblock3,
            self.rblock4,
            self.block2,
        ]:
            init_seq(b)

    def forward(self, uv, code):
        if self.code_dim > 0:
            x = self.block(F.leaky_relu(self.in_linear(uv) + code, 0.2))
        else:
            x = self.block(uv)
        x = x + self.rblock1(x)
        x = x + self.rblock2(x)
        x = x + self.rblock3(x)
        x = x + self.rblock4(x)
        x = self.block2(x)
        return x
