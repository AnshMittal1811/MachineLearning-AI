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

    def __init__(self, size_in, size_out=None, size_h=None):
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
        self.actvn = nn.ReLU()

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


class DisplacementDecoder(nn.Module):
    """DisplacementDecoder network class.

    It maps input points and time values together with (optional) conditioned
    codes c and latent codes z to the respective motion vectors.

    Args:
        in_dim (int): input dimension of points concatenated with the time axis
        out_dim (int): output dimension of motion vectors
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): size of the hidden dimension
        leaky (bool): whether to use leaky ReLUs as activation
        n_blocks (int): number of ResNet-based blocks
    """

    def __init__(
        self, in_dim=3, out_dim=3, c_dim=128, hidden_size=512, leaky=False, n_blocks=5, **kwargs
    ):
        super().__init__()
        self.c_dim = c_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_blocks = n_blocks
        # Submodules
        self.fc_p = nn.Linear(in_dim, hidden_size)

        self.fc_in = nn.Linear(c_dim * 2, c_dim)
        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, cur_t, fuc_t, c):
        batch_size, nsteps, c_dim = c.shape
        _, npoints, dim = p.shape
        cur_t = (
            torch.clamp(cur_t[:, :, None] * nsteps, 0, nsteps - 1)
            .expand(batch_size, 1, c_dim)
            .type(torch.LongTensor)
            .to(p.device)
        )
        fuc_t = (
            torch.clamp(fuc_t[:, :, None] * nsteps, 0, nsteps - 1)
            .expand(batch_size, 1, c_dim)
            .type(torch.LongTensor)
            .to(p.device)
        )
        cur_c = torch.gather(c, 1, cur_t)
        fuc_c = torch.gather(c, 1, fuc_t)
        # glo_c = torch.mean(c, dim=1)
        # concat_c = torch.cat([cur_c.squeeze(1), fuc_c.squeeze(1), glo_c], dim=1)
        concat_c = torch.cat([cur_c.squeeze(1), fuc_c.squeeze(1)], dim=1)
        concat_c = self.fc_in(concat_c)
        net = self.fc_p(p)

        # Layer loop
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net_c = self.fc_c[i](concat_c).unsqueeze(1)
                net = net + net_c
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        return out
