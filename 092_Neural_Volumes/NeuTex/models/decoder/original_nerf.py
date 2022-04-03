import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..networks import init_seq, positional_encoding


class OriginalNerfDecoder(nn.Module):
    def __init__(self, pos_freqs=10, view_freqs=4):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.view_freqs = view_freqs
        self.pos_size = 3 * pos_freqs * 2
        self.view_size = 3 * view_freqs * 2

        block1 = [
            nn.Linear(self.pos_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        ]
        self.block1 = nn.Sequential(*block1)

        block2 = [
            nn.Linear(256 + self.pos_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 257),
        ]
        self.block2 = nn.Sequential(*block2)

        block3 = [
            nn.Linear(256 + self.view_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softplus(),
        ]
        self.block3 = nn.Sequential(*block3)

        for b in [self.block1, self.block2, self.block3]:
            init_seq(b)

    def forward(self, pos, view_dir):
        """
            pos: `(*,3)`
            view_dir: `(*,3)`
        """
        ep = positional_encoding(pos, self.pos_freqs)
        ev = positional_encoding(view_dir, self.view_freqs)

        out1 = self.block2(torch.cat([self.block1(ep), ep], -1))
        density = F.softplus(out1[..., 0])
        color = self.block3(torch.cat([out1[..., 1:], ev], -1))

        return density, color
