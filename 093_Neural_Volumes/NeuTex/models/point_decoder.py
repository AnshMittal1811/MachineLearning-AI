import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import init_seq
from .base_decoder import BaseDecoder


# TODO: finish and test this class
class GaussianPointDecoder(BaseDecoder):
    def __init__(self, point_count, feature_dim=128, out_channels=4):
        super(GaussianPointDecoder, self).__init__()
        assert out_channels in [1, 4, 7, 8, 11], (
            "Output channels in decoder should be one of [1|4|7|8|11] to represent "
            "transmission(1), albedo(3), normal(3), roughness(1), and specular(3)"
        )
        self.out_channels = out_channels

        self.mu = nn.Parameter([point_count, 3])
        self.logstd = nn.Parameter([point_count, 3])
        self.features = nn.Parameter([point_count, 128])
        nn.init.uniform_(self.mu, -1, 1)
        nn.init.uniform_(self.logstd, -1, 5)
        nn.init.normal_(self.features, 0, 0.1)

        block = []
        in_channels = (
            self.mlp_channels * self.feature_freqs * 2
            + self.mlp_channels
            + 6 * self.position_freqs
        )
        in_channels = feature_dim
        out_channels = 128
        for i in range(5):
            block.append(nn.Linear(in_channels, out_channels))
            block.append(nn.LeakyReLU(0.2))
            in_channels = out_channels
            out_channels = 128
        self.block = nn.Sequential(*block)
        init_seq(self.block)

    def forward(self, pts, viewdirs=None, encoding=None, **kwargs):
        """
        Args:
            pts: :math:`(N, Rays, Samples, 3)`
        Return:
            output: standard output for decoder
        """

        std = torch.exp(self.logstd)
        f = (pts[:, :, :, None, :] - self.mu) / std
        f = 0.5 * (f * f).sum(-1)  # (N,Rays,Samples,Points)
        f = torch.exp(f) / std.prod(-1)  # pdf, (N,Rays,Samples,Points)

        features = (f[..., None] * self.features).sum(-2) / f.sum(
            -1, keepdim=True
        )  # (N,Rays,Samples,Features) / (N,Rays,Samples,1)

        output = self.block(features)
        # sigma
        output[..., [0]] = F.softplus(output[..., [0]])

        # albedo
        if self.out_channels >= 4:
            output[..., 1:4] = torch.sigmoid(output[..., 1:4])

        # normal
        if self.out_channels >= 7:
            output[..., 4:7] = torch.sigmoid(output[..., 4:7])
            output[..., 4:7] = F.normalize(2.0 * output[..., 4:7] - 1.0, dim=-1)

        # roughness
        if self.out_channels >= 8:
            output[..., 7] = torch.sigmoid(output[..., 7])

        # specular albedo
        if self.out_channels >= 11:
            output[..., 8:11] = torch.sigmoid(output[..., 8:11])

        return output

    def get_loss(self):
        return {"point_std": torch.mean(self.logstd ** 2)}
