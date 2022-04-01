from .oflow_point import ResnetPointnet
import torch
from torch import nn
from .lpdc_encoder import SpatioTemporalResnetPointnet


class GeometrySetEncoder(nn.Module):
    def __init__(self, frame_c_dim=128, frame_hidden_dim=256, hidden_dim=256, c_dim=128):
        super().__init__()
        self.frame_pointnet = ResnetPointnet(
            dim=3,
            c_dim=frame_c_dim,
            hidden_dim=frame_hidden_dim,
        )
        self.set_pointnet = ResnetPointnet(
            dim=frame_c_dim,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
        )
        return

    def forward(self, pc_seq):
        B, T, N, D = pc_seq.shape
        frame_pc = pc_seq.reshape(-1, N, D)
        frame_abs = self.frame_pointnet(frame_pc).reshape(B, T, -1)
        z_g = self.set_pointnet(frame_abs)
        return z_g


class ATCSetEncoder(nn.Module):
    def __init__(
        self,
        atc_num,
        c_dim,
        ci_dim,
        hidden_dim,
    ) -> None:
        super().__init__()
        self.backbone_pointnet = ResnetPointnet(
            dim=3,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
        )
        self.set_mlp_layers = nn.ModuleList(
            [nn.Linear(c_dim * 2, c_dim), nn.Linear(c_dim * 2, c_dim), nn.Linear(c_dim * 2, c_dim)]
        )
        self.theta_fc = nn.Linear(c_dim, atc_num)
        self.c_fc = nn.Sequential(nn.Linear(c_dim, c_dim), nn.ELU(), nn.Linear(c_dim, ci_dim))
        self.elu = nn.ELU()

    def forward(self, pc_set):
        B, T, N, D = pc_set.shape
        set_feature = self.backbone_pointnet(pc_set.reshape(-1, N, 3)).reshape(B, T, -1)  # B,T, C
        pooled = torch.max(set_feature, dim=1, keepdim=True).values
        pooled = pooled.expand(-1, T, -1)
        f = torch.cat([set_feature, pooled], dim=-1)
        for layer in self.set_mlp_layers:
            new_f = self.elu(layer(f))
            pooled_f = torch.max(new_f, dim=1, keepdim=True).values
            f = torch.cat([new_f, pooled_f.expand(-1, T, -1)], dim=-1)
        theta = self.theta_fc(new_f).reshape(B, T, -1)
        c_global = self.c_fc(pooled_f.squeeze(1))
        return c_global, theta


class ATCSetEncoder2(nn.Module):
    def __init__(
        self,
        atc_num,
        c_dim,
        ci_dim,
        hidden_dim,
    ) -> None:
        super().__init__()
        self.backbone_pointnet = ResnetPointnet(
            dim=3,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
        )
        self.set_mlp_layers = ResnetPointnet(
            dim=c_dim,
            c_dim=ci_dim,
            hidden_dim=hidden_dim,
        )
        self.theta_fc = nn.Linear(c_dim, atc_num)

    def forward(self, pc_set):
        B, T, N, D = pc_set.shape
        set_feature = self.backbone_pointnet(pc_set.reshape(-1, N, 3)).reshape(B, T, -1)  # B,T, C
        c_global, state_feat = self.set_mlp_layers(set_feature, return_unpooled=True)
        theta = self.theta_fc(state_feat)
        return c_global, theta


class Query1D(nn.Module):
    def __init__(self, ci_dim, c_dim, t_dim, amp_dim, hidden_dim) -> None:
        super().__init__()
        self.amp = nn.Sequential(
            nn.Linear(t_dim, amp_dim // 2), nn.LeakyReLU(), nn.Linear(amp_dim // 2, amp_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(c_dim + amp_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(c_dim, ci_dim), # ! warning, check this and fix this bug, c_dim != hidden dim
        )

    def forward(self, code, theta):
        # B,C; B,T,ATC
        B, T, _ = theta.shape
        amp = self.amp(theta)
        f = torch.cat([amp, code.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        ci = self.mlp(f)
        return ci


class Query1DLarger(nn.Module):
    def __init__(self, ci_dim, c_dim, t_dim, amp_dim, hidden_dim) -> None:
        super().__init__()
        self.amp = nn.Sequential(
            nn.Linear(t_dim, amp_dim // 2), nn.LeakyReLU(), nn.Linear(amp_dim // 2, amp_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(c_dim + amp_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, ci_dim),
        )

    def forward(self, code, theta):
        # B,C; B,T,ATC
        B, T, _ = theta.shape
        amp = self.amp(theta)
        f = torch.cat([amp, code.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        ci = self.mlp(f)
        return ci


# class LPDC_Encoder(nn.Module):
#     def __init__(self, c_dim, hidden_dim) -> None:
#         super().__init__()
#         self.lpdc_encoder = SpatioTemporalResnetPointnet(
#             c_dim=c_dim,
#             dim=3,
#             hidden_dim=hidden_dim,
#             use_only_first_pcl=False,
#             pool_once=False,
#         )
#         self.geometry_set_encoder = nn.Sequential(
#             nn.Linear(c_dim, c_dim), nn.LeakyReLU(), nn.Linear(c_dim, c_dim)
#         )

#     def forward(self, x):
#         return
