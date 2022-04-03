import torch
from torch import nn


class Time1DFeatureField(nn.Module):
    def __init__(
        self,
        c_dim=256,
        amp_dim=64,
        out_dim=256,
        hidden_dims=[256],
        use_bn=True,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        self.fc_t = nn.Conv1d(1, amp_dim, 1, 1, 0)
        c_in = c_dim + amp_dim
        layers = []
        for c_out in hidden_dims:
            layers.append(nn.Conv1d(c_in, c_out, 1, 1, 0))
            if use_bn:
                layers.append(nn.BatchNorm1d(c_out))
            layers.append(activation())
            c_in = c_out
        layers.append(nn.Conv1d(c_in, out_dim, 1, 1, 0))
        self.mlp = nn.Sequential(*layers)

    def forward(self, c_st, query_t):
        # c_st: B,C; query_t: B,T
        assert query_t.ndim == 2
        assert c_st.ndim == 2
        B, C = c_st.shape
        T = query_t.shape[1]
        t = query_t.unsqueeze(1)  # B,1,T
        net = self.fc_t(t)  # B,Amp,T
        code = c_st.unsqueeze(2).expand(B, C, T)
        net = torch.cat([net, code], dim=1)
        feat = self.mlp(net)
        return feat  # B,C,T
