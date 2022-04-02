import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, nf_v=1):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nf_v+3, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, v, f):
        B, M, C, H, W = f.size()
        f = f.contiguous().view(B*M, C, H, W)
        v = v.contiguous().view(B*M, v.size(2), 1, 1).repeat(1, 1, H, W)
        r = self.net(torch.cat((v, f), dim=1))
        
        return r