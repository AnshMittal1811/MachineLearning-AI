from core.net_bank.oflow_point import TemporalResnetPointnet
from core.net_bank.oflow_point import ResnetPointnet
import torch
from torch import nn


class TrajEncoder(nn.Module):
    def __init__(self, c_dim=128, hidden_dim=512, T=17) -> None:
        super().__init__()
        self.TNet = TemporalResnetPointnet(c_dim, T * 3, hidden_dim)
        self.GNet = ResnetPointnet(c_dim, 3, hidden_dim)
        self.last_fc = nn.Sequential(
            nn.Linear(2 * c_dim, 2 * c_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * c_dim, 2 * c_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * c_dim, c_dim),
        )

    def forward(self, x):
        B, T, N, _ = x.shape
        c_gt = self.TNet(x)
        c_g = self.GNet(x.reshape(B * T, N, 3)).reshape(B, T, -1)
        c = torch.cat([c_g, c_gt.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        ret = self.last_fc(c)
        return None, ret
