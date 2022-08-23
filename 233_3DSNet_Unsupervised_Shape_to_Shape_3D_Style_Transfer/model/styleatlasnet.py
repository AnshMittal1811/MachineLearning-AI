"""
Copyright (c) 2021, Mattia Segu
Licensed under the MIT License (see LICENSE for details)
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from model.model_blocks import Mapping2Dto3D, AdaptiveMapping2Dto3D, Identity, get_num_ada_norm_params
from model.template import get_template


class StyleAtlasnet(nn.Module):

    def __init__(self, opt):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        :param opt: 
        """
        super(StyleAtlasnet, self).__init__()
        self.opt = opt
        self.device = opt.device

        # Define number of points per primitives
        self.nb_pts_in_primitive = opt.number_points // opt.nb_primitives
        self.nb_pts_in_primitive_eval = opt.number_points_eval // opt.nb_primitives

        if opt.remove_all_batchNorms:
            torch.nn.BatchNorm1d = Identity
            print("Replacing all batchnorms by identities.")

        # Initialize templates
        self.template = [get_template(opt.template_type, device=opt.device) for i in range(0, opt.nb_primitives)]

        # Initialize deformation networks
        if opt.adaptive and opt.decode_style:
            self.decoder = nn.ModuleList([AdaptiveMapping2Dto3D(opt) for i in range(0, opt.nb_primitives)])

        else:
            self.decoder = nn.ModuleList([Mapping2Dto3D(opt) for i in range(0, opt.nb_primitives)])


    def forward(self, content_latent_vector, style_latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param train: a boolean indicating training mode
        :param content_latent_vector: an opt.bottleneck size vector encoding the content of a 3D shape.
                                      size : batch, bottleneck
        :param style_latent_vector: an opt.bottleneck size vector encoding the style of a 3D shape.
                                      size : batch, bottleneck
        :return: A deformed pointcloud of size : batch, nb_prim, num_point, 3
        """
        if train:
            input_points = [self.template[i].get_random_points(
                torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
                content_latent_vector.device) for i in range(self.opt.nb_primitives)]
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=content_latent_vector.device)
                            for i in range(self.opt.nb_primitives)]

        # Deform each patch
        if self.opt.adaptive and self.opt.decode_style:
            # in this case style latent vector represents the adabn parameters
            num_adabn_params = get_num_ada_norm_params(self.decoder[0])
            output_patches = [self.decoder[i](input_points[i],
                                              content_latent_vector.unsqueeze(2),
                                              style_latent_vector[:, i*num_adabn_params:(i+1)*num_adabn_params]
                                              ).unsqueeze(1)
                              for i in range(0, self.opt.nb_primitives)]
        else:
            output_patches = [self.decoder[i](input_points[i],
                                              content_latent_vector.unsqueeze(2),
                                              style_latent_vector.unsqueeze(2)).unsqueeze(1)
                              for i in range(0, self.opt.nb_primitives)]
        output_points = torch.cat(output_patches, dim=1)

        output = {
            'faces': None,
            # 'points_1': pred_y1,
            # 'points_2': pred_y2,
            'points_3': output_points.contiguous(),  # batch, nb_prim, num_point, 3
        }
        return output

    def generate_mesh(self, content_latent_vector, style_latent_vector, train=False):
        assert content_latent_vector.size(0) == 1, "input should have batch size 1!"
        import pymesh

        if train:
            input_points = [self.template[i].get_random_points(
                torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
                content_latent_vector.device) for i in range(self.opt.nb_primitives)]
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=content_latent_vector.device)
                            for i in range(self.opt.nb_primitives)]

        # input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive,
        #                                                     device=content_latent_vector.device)
        #                 for i in range(self.opt.nb_primitives)]
        input_points = [input_points[i] for i in range(self.opt.nb_primitives)]

        # Deform each patch
        if self.opt.adaptive and self.opt.decode_style:
            num_adabn_params = get_num_ada_norm_params(self.decoder[0])
            output_patches = [self.decoder[i](input_points[i],
                                              content_latent_vector.unsqueeze(2),
                                              style_latent_vector[:, i*num_adabn_params:(i+1)*num_adabn_params]
                                              ).unsqueeze(1)
                              for i in range(0, self.opt.nb_primitives)]
        else:
            output_patches = [self.decoder[i](input_points[i],
                                              content_latent_vector.unsqueeze(2),
                                              style_latent_vector.unsqueeze(2)).unsqueeze(1)
                              for i in range(0, self.opt.nb_primitives)]
        output_points = torch.cat(output_patches, dim=1).squeeze(0)

        output_meshes = [pymesh.form_mesh(vertices=output_points[i].transpose(1, 0).contiguous().cpu().numpy(),
                                          faces=self.template[i].mesh.faces)
                         for i in range(self.opt.nb_primitives)]

        # Deform return the deformed pointcloud
        mesh = pymesh.merge_meshes(output_meshes)

        return mesh

