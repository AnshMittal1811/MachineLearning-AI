import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import logging

def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices
    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1] ** 2
    yy = quaternions[..., 2] ** 2
    zz = quaternions[..., 3] ** 2
    ww = quaternions[..., 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask)  # 1,1,1,3

    def forward(self, F, y):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x1)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


class SimpleNVP(nn.Module):
    def __init__(
        self,
        n_layers,
        feature_dims,
        hidden_size,
        projection,
        checkpoint=True,
        normalize=True,
        explicit_affine=True,
    ):
        super().__init__()
        self._checkpoint = checkpoint
        self._normalize = normalize
        self._explicit_affine = explicit_affine
        self._projection = projection
        self._create_layers(n_layers, feature_dims, hidden_size)

    def _create_layers(self, n_layers, feature_dims, hidden_size):
        input_dims = 3
        proj_dims = self._projection.proj_dims

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(input_dims)
            # mask[torch.arange(input_dims) % 2 == (i%2)] = 1
            mask[torch.randperm(input_dims)[:2]] = 1
            logging.info("NVP {}th layer split is {}".format(i,mask))

            map_s = nn.Sequential(
                nn.Linear(proj_dims + feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
                nn.Hardtanh(min_val=-10, max_val=10),
            )
            map_t = nn.Sequential(
                nn.Linear(proj_dims + feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
            )
            self.layers.append(
                CouplingLayer(map_s, map_t, self._projection, mask[None, None, None])
            )

        if self._normalize:
            self.scales = nn.Sequential(
                nn.Linear(feature_dims, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 3)
            )

        if self._explicit_affine:
            self.rotations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 4)
            )
            self.translations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 3)
            )

    def _check_shapes(self, F, x):
        B1, M1, _ = F.shape  # batch, templates, C
        B2, _, M2, D = x.shape  # batch, Npts, templates, 3
        assert B1 == B2 and M1 == M2 and D == 3

    def _expand_features(self, F, x):
        _, N, _, _ = x.shape
        return F[:, None].expand(-1, N, -1, -1)

    def _call(self, func, *args, **kwargs):
        if self._checkpoint:
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _normalize_input(self, F, y):
        if not self._normalize:
            return 0, 1

        sigma = torch.nn.functional.elu(self.scales(F)) + 1
        sigma = sigma[:, None]

        return 0, sigma

    def _affine_input(self, F, y):
        if not self._explicit_affine:
            return torch.eye(3)[None, None, None].to(F.device), 0

        q = self.rotations(F)
        q = q / torch.sqrt((q ** 2).sum(-1, keepdim=True))
        R = quaternions_to_rotation_matrices(q[:, None])
        t = self.translations(F)[:, None]

        return R, t

    def forward(self, F, x):
        self._check_shapes(F, x)
        mu, sigma = self._normalize_input(F, x)
        R, t = self._affine_input(F, x)
        F = self._expand_features(F, x)

        y = x
        ldj = 0
        for l in self.layers:
            y, ldji = self._call(l, F, y)
            ldj = ldj + ldji
        y = y / sigma + mu
        y = torch.matmul(y.unsqueeze(-2), R).squeeze(-2) + t
        return y

    def inverse(self, F, y):
        self._check_shapes(F, y)
        mu, sigma = self._normalize_input(F, y)
        R, t = self._affine_input(F, y)
        F = self._expand_features(F, y)

        x = y
        x = torch.matmul((x - t).unsqueeze(-2), R.transpose(-2, -1)).squeeze(-2)
        x = (x - mu) * sigma
        ldj = 0
        for l in reversed(self.layers):
            x, ldji = self._call(l.inverse, F, x)
            ldj = ldj + ldji
        return x, ldj
