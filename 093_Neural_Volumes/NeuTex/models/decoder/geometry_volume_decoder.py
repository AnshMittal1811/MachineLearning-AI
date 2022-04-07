import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..networks import init_seq


class GeometryVolumeDecoder(nn.Module):
    def __init__(
        self,
        template_class,
        input_dim,
        uv_dim=0,
        requested_features={"density", "normal"},
        template_res=128,
    ):
        super().__init__()

        assert input_dim > 0
        assert uv_dim in [0, 2, 3, 4]

        self.requested_features = requested_features

        self.uv_dim = uv_dim
        self.template_res = template_res

        self.out_channels = 0
        if "density" in self.requested_features:
            self.out_channels += 1
        if "normal" in self.requested_features:
            self.out_channels += 3
        if "uv" in self.requested_features:
            self.out_channels += self.uv_dim

        self.template = template_class(input_dim, self.out_channels, template_res)

    def forward(self, latent_code, pts, viewdirs=None):
        """
        Args:
            latent_code: :math:`(N,Code)`
            pts: :math:`(N,Rays,Samples,3)`
        """
        output = self.template(latent_code, pts)

        result = {}

        index = 0
        if "density" in self.requested_features:
            result["density"] = F.softplus(output[..., index])
            index += 1
        if "normal" in self.requested_features:
            result["normal"] = F.normalize(
                torch.tanh(output[..., index : index + 3]), dim=-1
            )
            index += 3
        if "uv" in self.requested_features:
            result["uv"] = F.tanh(output[..., index:])

        return result


# class GeometryVolumeDecoder(nn.Module):
#     def __init__(self, template_class, input_dim, uv_dim, template_res=128):
#         super().__init__()

#         assert input_dim > 0
#         assert uv_dim in [0, 2, 3, 4]

#         self.uv_dim = uv_dim

#         self.out_channels = 1 + uv_dim + 4
#         self.template_res = template_res
#         self.template = template_class(input_dim, self.out_channels, template_res)

#     def forward(self, input_encoding, pts, viewdirs=None):
#         """
#         Args:
#             input_encoding: :math:`(N,Rays)`
#             pts: :math:`(N,Rays,Samples,3)`
#         """
#         output = self.template(input_encoding, pts)

#         sigma = F.softplus(output[..., 0])
#         if self.uv_dim == 0:
#             uv = None
#         else:
#             uv = torch.tanh(output[..., 1 : 1 + self.uv_dim])
#         quat = torch.tanh(output[..., 1 + self.uv_dim : 5 + self.uv_dim])
#         quat = F.normalize(quat, dim=-1)

#         return sigma, uv, quat


class GeometryVolumeMlpDecoder(nn.Module):
    def __init__(
        self,
        template_class,
        input_dim,
        uv_dim,
        uv_count,
        template_output_dim,
        template_res=128,
    ):
        super().__init__()

        assert input_dim > 0
        assert uv_dim in [0, 2, 3, 4]
        assert uv_count > 0

        self.template_output_dim = template_output_dim
        self.uv_dim = uv_dim
        self.uv_count = uv_count

        if self.uv_count == 1:
            self.out_channels = 1 + 4 + self.uv_dim
        else:
            self.out_channels = 1 + 4 + self.uv_dim * self.uv_count + self.uv_count

        self.template_res = template_res
        self.template = template_class(input_dim, template_output_dim, template_res)

        block = []
        block.append(nn.Linear(template_output_dim + 3, 128))
        block.append(nn.LeakyReLU(0.2))
        for i in range(2):
            block.append(nn.Linear(128, 128))
            block.append(nn.LeakyReLU(0.2))
        block.append(nn.Linear(128, self.out_channels))
        self.block = nn.Sequential(*block)
        init_seq(self.block)

    def forward(self, input_encoding, pts, viewdirs=None):
        """
        Args:
            input_encoding: :math:`(N,Rays)`
            pts: :math:`(N,Rays,Samples,3)`
        """
        output = self.template(input_encoding, pts)
        output = self.block(torch.cat((output, pts), -1))

        sigma = F.softplus(output[..., 0])
        quat = torch.tanh(output[..., 1:5])
        quat = F.normalize(quat, dim=-1)

        uv_weights = None
        if self.uv_dim == 0:
            uv = None
        elif self.uv_count == 1:
            uv = torch.tanh(output[..., 5:])
        else:
            uv = torch.tanh(output[..., 5 : 5 + self.uv_dim * self.uv_count])
            uv = uv.view(uv.shape[:-1] + (self.uv_count, self.uv_dim))
            uv_weights = F.softmax(output[..., 5 + self.uv_dim * self.uv_count :], -1)

        return sigma, quat, uv, uv_weights
