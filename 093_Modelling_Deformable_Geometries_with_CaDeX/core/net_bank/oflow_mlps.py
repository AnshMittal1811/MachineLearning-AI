# clean mlps from occupancy flow
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, actvn=F.relu):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = actvn

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class FlowMLP(nn.Module):
    def __init__(
        self,
        in_dim=4,
        out_dim=3,
        feature_dim=128,
        hidden_size=128,
        actvn=F.softplus,
        n_blocks=5,
        zero_init_last=False,
    ):
        super().__init__()
        self.f_dim = feature_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_blocks = n_blocks
        self.actvn = actvn
        # Submodules
        self.fc_p = nn.Linear(in_dim, hidden_size)
        self.fc_f = nn.ModuleList([nn.Linear(self.f_dim, hidden_size) for i in range(n_blocks)])

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size, actvn=self.actvn) for i in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_size, self.out_dim)

        if zero_init_last:
            self.fc_out.weight.data.fill_(0.0)
            self.fc_out.bias.data.fill_(0.0)

    def forward(self, query, feature, t=None):
        n_batch, n_pts, _ = query.shape
        if feature.ndim == 2:
            feature = feature.unsqueeze(1).repeat(1, n_pts, 1)
        assert feature.shape[1] == n_pts
        feature = feature.reshape(n_batch * n_pts, -1)
        if t is not None:
            if t.ndim == 1:  # T
                t = t.unsqueeze(1).unsqueeze(1).repeat(1, n_pts, 1)
            elif t.ndim == 2:  # B,T
                t = t.unsqueeze(1).repeat(1, n_pts, 1)
            coordinate = torch.cat([query, t], dim=2).reshape(-1, self.in_dim)
        else:
            coordinate = query.reshape(-1, self.in_dim)
        net = self.fc_p(coordinate)

        # Layer loop
        for i in range(self.n_blocks):
            net_c = self.fc_f[i](feature)
            net = net + net_c
            net = self.blocks[i](net)
        flow = self.fc_out(self.actvn(net)).reshape(n_batch, n_pts, self.out_dim)
        return flow


class OccMLP(nn.Module):
    def __init__(
        self,
        in_dim=3,
        out_dim=3,
        feature_dim=128,
        hidden_size=128,
        actvn=F.softplus,
        n_blocks=5,
        zero_init_last=False,
    ):
        super().__init__()
        self.f_dim = feature_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_blocks = n_blocks
        self.actvn = actvn
        # Submodules
        self.fc_p = nn.Linear(in_dim, hidden_size)
        self.fc_f = nn.Linear(self.f_dim, hidden_size)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size, actvn=self.actvn) for i in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_size, self.out_dim)

        if zero_init_last:
            self.fc_out.weight.data.fill_(0.0)
            self.fc_out.bias.data.fill_(0.0)

    def forward(self, query, feature):
        n_batch, n_pts, in_dim = query.shape
        assert in_dim == self.in_dim
        if feature.ndim == 2:
            feature = feature.unsqueeze(1).repeat(1, n_pts, 1)
        assert feature.shape[1] == n_pts
        feature = feature.reshape(n_batch * n_pts, -1)
        coordinate = query.reshape(-1, self.in_dim)
        net = self.fc_p(coordinate)
        net_c = self.fc_f(feature)
        net = net + net_c
        # Layer loop
        for i in range(self.n_blocks):
            net = self.blocks[i](net)
        occ = self.fc_out(self.actvn(net)).reshape(n_batch, n_pts, self.out_dim)
        return occ


if __name__ == "__main__":
    net1 = FlowMLP(
        in_dim=4,
        out_dim=3,
        feature_dim=128,
        hidden_size=128,
        actvn=F.softplus,
        n_blocks=5,
        zero_init_last=True,
    ).cuda()
    net2 = OccMLP(
        in_dim=3,
        out_dim=1,
        feature_dim=128,
        hidden_size=128,
        actvn=F.softplus,
        n_blocks=5,
        zero_init_last=True,
    ).cuda()
    net3 = FlowMLP(
        in_dim=3,
        out_dim=3,
        feature_dim=128,
        hidden_size=128,
        actvn=F.softplus,
        n_blocks=5,
        zero_init_last=True,
    ).cuda()
    f, q, t = torch.rand(2, 1024, 128).cuda(), torch.rand(2, 1024, 3).cuda(), torch.rand(2).cuda()
    flow = net1(q, f, t)
    print(flow.shape)
    print(flow.sum())
    flow2 = net3(q, f)
    print(flow2.shape)
    print(flow2.sum())
    occ = net2(q, f)
    print(occ.shape)
    print(occ.sum())