import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..networks import init_seq


class StandardVolumeDecoder(nn.Module):
    def __init__(self, template_class, input_dim, out_channels=4, template_res=128):
        super(StandardVolumeDecoder, self).__init__()
        self.template_res = template_res
        self.out_channels = out_channels
        self.template = template_class(input_dim, out_channels, template_res)

    def forward(self, input_encoding, pts, viewdirs=None):
        """
        Args:
            pts: :math:`(N,Rays,Samples,3)`
        """

        output = self.template(input_encoding, pts)

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

        return {"output": output}
