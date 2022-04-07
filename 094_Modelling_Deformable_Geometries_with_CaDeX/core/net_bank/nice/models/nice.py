import torch
from torch import masked_select, nn
from torch.utils.checkpoint import checkpoint
import logging
from .projection_layer import get_projection_layer


class CouplingLayer(nn.Module):
    def __init__(self, map_t, projection, mask):
        super().__init__()
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask)  # 1,1,1,3

    def forward(self, F, y):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1)], dim=-1)
        t = self.map_t(F_y1)

        x = y1 + (1 - self.mask) * (y - t)

        return x, None

    def inverse(self, F, x):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x1)], dim=-1)
        t = self.map_t(F_x1)

        y = x1 + (1 - self.mask) * (x + t)

        return y, None


class MLP(nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=nn.LeakyReLU, bn=nn.BatchNorm1d):
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(nn.Conv1d(d_in, d_out, 1, 1, 0))
            if bn is not None:
                layers.append(bn(d_out))
            layers.append(act())
            d_in = d_out
        layers.append(nn.Conv1d(d_in, c_out, 1, 1, 0))
        self.mlp = nn.Sequential(*layers)
        self.c_out = c_out

    def forward(self, x):
        # x: B,...,C_in
        input_shape = x.shape
        B, C = input_shape[0], input_shape[-1]
        _x = x.reshape(B, -1, C).transpose(2, 1)  # B,C_in,X
        y = self.mlp(_x)  # B,C_out, X
        y = y.transpose(2, 1).reshape(*input_shape[:-1], self.c_out)
        return y


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


class NICE(nn.Module):
    # * NICE is from NVP_v2_2
    def __init__(
        self,
        n_layers,
        feature_dims,
        hidden_size,
        proj_dims,
        code_proj_hidden_size=[],
        proj_type="simple",
        block_normalize=False,
        normalization=False,
        explicit_affine=False,
    ):
        super().__init__()
        self._checkpoint = False
        self._normalize = block_normalize
        self._explicit_affine = explicit_affine

        # make layers
        input_dims = 3
        normalization = nn.InstanceNorm1d if normalization else None

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.code_projectors = nn.ModuleList()
        self.layer_idx = [i for i in range(n_layers)]

        i = 0
        mask_selection = []
        while i < n_layers:
            mask_selection.append(torch.randperm(input_dims))
            i += input_dims
        mask_selection = torch.cat(mask_selection)
        logging.info("NVP-v2 mask selection {}".format(mask_selection))

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        for i in self.layer_idx:

            # get mask
            mask2 = torch.zeros(input_dims)
            mask2[mask_selection[i]] = 1
            mask1 = 1 - mask2
            logging.info("NVP-v2 {}th block splits are {}+{}".format(i, mask1, mask2))

            # get z transform
            map_t = MLP(proj_dims + feature_dims, input_dims, hidden_size, bn=normalization)
            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type)
            self.layers1.append(CouplingLayer(map_t, proj, mask1[None, None, None]))

            # get xy transform (tiny)
            map_t = MLP(proj_dims + feature_dims, input_dims, hidden_size[:1], bn=normalization)
            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type)
            self.layers2.append(CouplingLayer(map_t, proj, mask2[None, None, None]))

            # get code projector
            if len(code_proj_hidden_size) == 0:
                code_proj_hidden_size = [feature_dims]
            self.code_projectors.append(
                MLP(feature_dims, feature_dims, code_proj_hidden_size, bn=normalization)
            )

        if self._normalize:
            self.scales = nn.Sequential(
                nn.Linear(feature_dims, hidden_size[0]), nn.ReLU(), nn.Linear(hidden_size[0], 3)
            )

        if self._explicit_affine:
            self.rotations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size[0]), nn.ReLU(), nn.Linear(hidden_size[0], 4)
            )
            self.translations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size[0]), nn.ReLU(), nn.Linear(hidden_size[0], 3)
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
        # F: B,N,T,C x: B,N,T,3
        y = x
        y = torch.matmul(y.unsqueeze(-2), R).squeeze(-2) + t
        for i in self.layer_idx:
            # get block condition code
            Fi = self.code_projectors[i](F)
            Fi = self._expand_features(Fi, y)
            # first transformation
            l1 = self.layers1[i]
            y, _ = self._call(l1, Fi, y)
            # second transformation
            l2 = self.layers2[i]
            y, _ = self._call(l2, Fi, y)
        y = y / sigma + mu
        return y

    def inverse(self, F, y):
        self._check_shapes(F, y)
        mu, sigma = self._normalize_input(F, y)
        R, t = self._affine_input(F, y)

        x = y
        x = (x - mu) * sigma
        ldj = 0
        for i in reversed(self.layer_idx):
            # get block condition code
            Fi = self.code_projectors[i](F)
            Fi = self._expand_features(Fi, x)
            # reverse second transformation
            l2 = self.layers2[i]
            x, _ = self._call(l2.inverse, Fi, x)
            # ldj = ldj + ldji
            # reverse first transformation
            l1 = self.layers1[i]
            x, _ = self._call(l1.inverse, Fi, x)
            # ldj = ldj + ldji
        x = torch.matmul((x - t).unsqueeze(-2), R.transpose(-2, -1)).squeeze(-2)
        return x, ldj
