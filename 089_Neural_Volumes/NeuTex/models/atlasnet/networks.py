import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..networks import init_weights, init_seq

"""
modified from
https://github.com/ThibaultGROUEIX/AtlasNet
"""


class Mapping2Dto3D(nn.Module):
    """
    Modified AtlasNet core function
    """

    def __init__(
        self,
        code_size,
        input_point_dim,
        hidden_size=512,
        num_layers=2,
        activation="relu",
    ):
        """
        template_size: input size
        """
        super().__init__()
        assert activation in ["relu", "softplus"]

        self.code_size = code_size
        self.input_size = input_point_dim
        self.dim_output = 3
        self.hidden_neurons = hidden_size
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.input_size, self.code_size)
        self.linear2 = nn.Linear(self.code_size, self.hidden_neurons)

        init_weights(self.linear1)
        init_weights(self.linear2)

        self.linear_list = nn.ModuleList(
            [
                nn.Linear(self.hidden_neurons, self.hidden_neurons)
                for i in range(self.num_layers)
            ]
        )

        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(self.hidden_neurons, self.dim_output)
        init_weights(self.last_linear)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "softplus":
            self.activation = F.softplus

        # self.diff_geom = DiffGeomProps()

    def _forward(self, x, latent):
        x = self.linear1(x) + latent
        x = self.activation(x)
        x = self.activation(self.linear2(x))
        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        return self.last_linear(x)

    def forward(self, x, latent):
        assert x.shape[-1] == self.input_size
        self.uv_ = x

        self.xyz_ = self._forward(x, latent)
        return self.xyz_

    def compute_normal(self, x, latent, eps=0.01):
        assert x.shape[-1] == self.input_size
        with torch.no_grad():
            angles = torch.rand(x.shape[:-1], dtype=x.dtype).to(x.device) * np.pi * 2
            _x = torch.cos(angles)
            _y = torch.sin(angles)
            dir1 = torch.stack([_x, _y], dim=-1)
            dir2 = torch.stack([-_y, _x], dim=-1)

        if self.input_size == 2:
            x1 = x + dir1 * eps
            x2 = x + dir2 * eps

        elif self.input_size == 3:
            with torch.no_grad():
                v1 = torch.zeros_like(x, dtype=x.dtype).to(x.device)
                v2 = torch.zeros_like(x, dtype=x.dtype).to(x.device)
                v1[..., 2] = 1
                v2[..., 0] = 1

                c1 = x.cross(v1)
                c2 = x.cross(v2)
                mask = (c1 * c1).sum(-1, keepdim=True) > (c2 * c2).sum(-1, keepdim=True)
                mask = mask.float()
                t1 = c1 * mask + c2 * (1 - mask)
                t1 = F.normalize(t1, dim=-1)
                t2 = x.cross(t1)
                d1 = t1 * dir1[..., [0]] + t2 * dir1[..., [1]]
                d2 = t1 * dir2[..., [0]] + t2 * dir2[..., [1]]

                x1 = F.normalize(x + d1 * eps, dim=-1)
                x2 = F.normalize(x + d2 * eps, dim=-1)

        p = self._forward(x, latent)
        p1 = self._forward(x1, latent)
        p2 = self._forward(x2, latent)
        normal = F.normalize((p1 - p).cross(p2 - p), dim=-1)

        return p, normal


class SquareTemplate:
    def __init__(self):
        self.regular_num_points = 0

    def get_random_points(self, npoints, device):
        with torch.no_grad():
            rand_grid = (torch.rand((npoints, 2)) * 2 - 1).to(device).float()
            return rand_grid

    def get_regular_points(self, npoints=2500, device=None):
        """
        Get regular points on a Square
        """
        assert int(npoints ** 0.5) ** 2 == npoints
        assert device is not None, "device needs to be provided for get_regular_points"

        side_length = int(npoints ** 0.5)

        uv = np.stack(
            np.meshgrid(*([np.linspace(-1, 1, side_length)] * 2), indexing="ij"),
            axis=-1,
        ).reshape((-1, 2))

        points = torch.FloatTensor(uv).to(device)
        return points.requires_grad_()


class SphereTemplate:
    def get_random_points(self, npoints, device):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        with torch.no_grad():
            points = torch.randn((npoints, 3)).to(device).float() * 2 - 1
            points = F.normalize(points, dim=-1)
        return points

    def get_regular_points(self, npoints, device):
        """
        Get regular points on a Sphere
        Return Tensor of Size [x, 3]
        """
        import trimesh
        mesh = trimesh.creation.icosphere(6)
        return torch.tensor(mesh.vertices).to(device).float()


class Atlasnet(nn.Module):
    def __init__(
        self,
        num_points_per_primitive,
        num_primitives,
        code_size,
        activation,
        primitive_type="square",
    ):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt: 
        """
        super().__init__()

        if primitive_type == "square":
            self.template_class = SquareTemplate
            self.input_point_dim = 2
        elif primitive_type == "sphere":
            self.template_class = SphereTemplate
            self.input_point_dim = 3
        else:
            raise Exception("Unknown primitive type {}".format(primitive_type))

        self.num_points_per_primitive = num_points_per_primitive
        self.num_primitives = num_primitives

        # Initialize templates
        self.template = [self.template_class() for i in range(0, num_primitives)]

        # Intialize deformation networks
        self.decoder = nn.ModuleList(
            [
                Mapping2Dto3D(code_size, self.input_point_dim, activation=activation)
                for i in range(0, num_primitives)
            ]
        )

        with torch.no_grad():
            self.label = torch.zeros(num_points_per_primitive * num_primitives).long()
            for i in range(num_primitives):
                self.label[
                    num_points_per_primitive * i : num_points_per_primitive * (i + 1)
                ] = i

    def get_label(self, device):
        self.label = self.label.to(device)
        return self.label

    def forward(self, latent_vector, regular_point_count=None):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """

        if regular_point_count is None:
            input_points = [
                self.template[i].get_random_points(
                    self.num_points_per_primitive, latent_vector.device,
                )
                for i in range(self.num_primitives)
            ]
        else:
            input_points = [
                self.template[i].get_regular_points(
                    regular_point_count, latent_vector.device
                )
                for i in range(self.num_primitives)
            ]

        points2d = [
            self.decoder[i](
                input_points[i].unsqueeze(0), latent_vector.unsqueeze(1)
            ).unsqueeze(1)
            for i in range(0, self.num_primitives)
        ]

        output_points = torch.cat(points2d, dim=1)  # batch, nb_prim, num_point, 3

        return input_points, output_points.contiguous()

    def map(self, latent_vector, uvs):
        """
        uvs: (N,...,P,2/3)
        latent_vector: (N,V)
        """
        assert uvs.shape[-1] == self.input_point_dim
        assert uvs.shape[-2] == self.num_primitives
        input_shape = uvs.shape
        output = torch.stack(
            [
                self.decoder[i](
                    uvs[..., i, :].view(input_shape[0], -1, self.input_point_dim),
                    latent_vector.unsqueeze(1),
                )
                for i in range(self.num_primitives)
            ],
            dim=-2,
        )
        return output.view(input_shape[:-1] + (3,))

    def map_and_normal(self, latent_vector, uvs, eps=0.01):
        assert uvs.shape[-1] == self.input_point_dim
        assert uvs.shape[-2] == self.num_primitives
        input_shape = uvs.shape
        outputs = []
        normals = []
        for i in range(self.num_primitives):
            output, normal = self.decoder[i].compute_normal(
                uvs[..., i, :].view(input_shape[0], -1, self.input_point_dim),
                latent_vector.unsqueeze(1),
                eps=eps,
            )
            outputs.append(output)
            normals.append(normal)

        outputs = torch.stack(outputs, dim=-2).view(input_shape[:-1] + (3,))
        normals = torch.stack(normals, dim=-2).view(input_shape[:-1] + (3,))
        return outputs, normals
