import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class Shader(nn.Module):
    # adapted from nerf
    # https://github.com/yenchenlin/nerf-pytorch/blob/bdb012ee7a217bfd96be23060a51df7de402591e/run_nerf_helpers.py#L67
    def __init__(self, feat_dim=256, output_opacity=False, opacity_channel=2, rgb_channel=3, basis_type='mlp'):
        self.SH_basis_dim = 9

        self.output_channel = rgb_channel
        super(Shader, self).__init__()
        self.feat_dim = feat_dim
        self.basis_type = basis_type

        if basis_type=='mlp':
            embeddirs_fn, input_ch_views = get_embedder(multires=4, i=0)

            self.input_ch_views = input_ch_views
            self.embeddirs_fn = embeddirs_fn

            self.opacity_output_channel = 0
            if output_opacity:
                self.opacity_output_channel += opacity_channel # save a channel for radius

            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + feat_dim, feat_dim // 2)])
            # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + feat_dim, feat_dim // 2), nn.Linear(feat_dim//2, feat_dim//2)])
            self.rgb_linear = nn.Linear(feat_dim // 2, self.output_channel)

            if self.opacity_output_channel > 0:
                self.view_independent_linear = nn.ModuleList([nn.Linear(feat_dim, feat_dim // 2)]) # this should not depend on view
                self.opacity_linear = nn.Linear(feat_dim // 2, self.opacity_output_channel)

        elif basis_type=='SH':
            # forward input: N x rgb_channel x SH_basis_dim
            assert (not output_opacity) # not supported
            assert feat_dim == self.SH_basis_dim * rgb_channel

        elif basis_type=='none':
            assert (not output_opacity)
            assert feat_dim == rgb_channel
            # do nothing

        else:
            raise NotImplementedError

    def forward(self, feature, input_views):
        # feature shold have shape B x W
        # input_views should have shape B x 3

        if self.basis_type=='mlp':
            B, C = feature.shape
            assert (C==self.feat_dim)
            _, C1 = input_views.shape
            assert (C1 == 3)

            embedded_views = self.embeddirs_fn(input_views)

            h = torch.cat([feature, embedded_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)

            if self.opacity_output_channel > 0:
                for i, l in enumerate(self.view_independent_linear):
                    feature = self.view_independent_linear[i](feature)
                    feature = F.relu(feature)

                opacity = self.opacity_linear(feature)
                # opacity = torch.sigmoid(opacity) # opacity and radius should be positive

                return torch.cat([rgb, opacity], dim=-1) # B x 3or5

            else:
                return rgb

        elif self.basis_type=='SH':
            # forward input: N x (rgb_channel*SH_basis_dim)
            feature = feature.reshape(-1, self.output_channel, self.SH_basis_dim)
            sh_mult = eval_sh_bases(self.SH_basis_dim, input_views) # N x 9
            rgb = torch.sum(sh_mult.unsqueeze(-2) * feature, dim=-1) # N x rgb_channel

            return rgb

        elif self.basis_type=='none':
            # forward input: N x rgb_channel xSH_basis_dim
            return feature

        else:
            raise NotImplementedError

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

# adapted from https://github.com/sxyu/svox2/blob/59984d6c4fd3d713353bafdcb011646e64647cc7/svox2/utils.py#L115
def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return result