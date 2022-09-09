"""
Copyright (c) 2021, Mattia Segu
Licensed under the MIT License (see LICENSE for details)
"""

from model.atlasnet import Atlasnet
from model.styleatlasnet import StyleAtlasnet
from model.model_blocks import PointNet, MLP, get_num_ada_norm_params
import model.resnet as resnet
from model.meshflow import NeuralMeshFlow
import torch
import torch.nn as nn
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes


class StyleNetBase(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    """

    def __init__(self, opt):
        super(StyleNetBase, self).__init__()

        self.classes = opt.class_choice
        self.classes = [c.replace('.', ' ') for c in self.classes]
        self.sample_points_from_mesh = opt.multiscale_loss
        self.SVR_0 = opt.SVR_0
        self.SVR_1 = opt.SVR_1
        self.type_0 = 'image' if opt.SVR_0 else 'points'
        self.type_1 = 'image' if opt.SVR_1 else 'points'
        self.decoder_type = opt.decoder_type

        if self.decoder_type.lower() == "atlasnet":
            DecoderArchitecture = StyleAtlasnet
        elif self.decoder_type.lower() == "meshflow":
            DecoderArchitecture = NeuralMeshFlow
        else:
            raise IOError("Type of decoder currently not supported.")

        self.style_bottleneck_size = opt.style_bottleneck_size if opt.adaptive else opt.hidden_neurons
        self.cycle_reconstruction = True if opt.weight_cycle_chamfer > 0 else False

        # Define content encoders
        if opt.SVR_0:
            content_encoder_0 = resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)
        else:
            content_encoder_0 = PointNet(nlatent=opt.bottleneck_size, normalization=opt.generator_norm)
        if opt.SVR_1:
            if opt.share_content_encoder and opt.SVR_0:
                content_encoder_1 = content_encoder_0
            else:
                content_encoder_1 = resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)
        else:
            if opt.share_content_encoder and not opt.SVR_0:
                content_encoder_1 = content_encoder_0
            else:
                content_encoder_1 = PointNet(nlatent=opt.bottleneck_size, normalization=opt.generator_norm)

        self.content_encoder = {self.classes[0]: content_encoder_0,
                                self.classes[1]: content_encoder_1}
        self.content_encoder = nn.ModuleDict(self.content_encoder)

        # Define decoders
        decoder_0 = DecoderArchitecture(opt)
        decoder_1 = decoder_0 if opt.share_decoder else DecoderArchitecture(opt)
        self.decoder = {self.classes[0]: decoder_0,
                        self.classes[1]: decoder_1}
        self.decoder = nn.ModuleDict(self.decoder)

        # Define style encoders
        if opt.SVR_0:
            style_encoder_0 = resnet.resnet18(pretrained=False, num_classes=opt.style_bottleneck_size)
        else:
            style_encoder_0 = PointNet(nlatent=opt.style_bottleneck_size, normalization=opt.generator_norm)
        if opt.SVR_1:
            if opt.share_style_encoder and opt.SVR_0:
                style_encoder_1 = style_encoder_0
            else:
                style_encoder_1 = resnet.resnet18(pretrained=False, num_classes=opt.style_bottleneck_size)
        else:
            if opt.share_style_encoder and not opt.SVR_0:
                style_encoder_1 = style_encoder_0
            else:
                style_encoder_1 = PointNet(nlatent=opt.style_bottleneck_size, normalization=opt.generator_norm)

        if opt.adaptive and opt.decode_style:
            if self.decoder_type.lower() == "atlasnet":
                self.num_ada_norm_params = [get_num_ada_norm_params(self.decoder[self.classes[0]].decoder[i])
                                         for i in range(0, opt.nb_primitives)]
                total_ada_norm_params = sum(self.num_ada_norm_params)
            elif self.decoder_type.lower() == "meshflow":
                self.num_ada_norm_params = get_num_ada_norm_params(self.decoder[self.classes[0]])
                total_ada_norm_params = self.num_ada_norm_params
            else:
                raise IOError("Type of decoder currently not supported.")

            self.style_mlp = {}
            style_mlp_0 = MLP(opt.style_bottleneck_size, total_ada_norm_params, opt.hidden_neurons,
                              opt.num_layers_mlp, normalization='none', rate=0.3, activation='relu')
            style_mlp_1 = style_mlp_0 if (opt.share_style_encoder or opt.share_style_mlp) else MLP(
                opt.style_bottleneck_size, total_ada_norm_params, opt.hidden_neurons, opt.num_layers_mlp,
                normalization='none', rate=0.3, activation='relu')
            self.style_encoder = {self.classes[0]: nn.Sequential(style_encoder_0, style_mlp_0),
                                  self.classes[1]: nn.Sequential(style_encoder_1, style_mlp_1)}
        else:
            self.style_encoder = {self.classes[0]: style_encoder_0,
                                  self.classes[1]: style_encoder_1}

        self.style_encoder = nn.ModuleDict(self.style_encoder)

        # Define discriminator
        discriminator_encoder_0 = PointNet(nlatent=opt.dis_bottleneck_size,
                                           normalization=opt.discriminator_norm,
                                           activation=opt.discriminator_activation)
        discriminator_encoder_1 = discriminator_encoder_0 if opt.share_discriminator_encoder else PointNet(
            nlatent=opt.dis_bottleneck_size,
            normalization=opt.discriminator_norm,
            activation=opt.discriminator_activation)
        self.discriminator_encoder = {self.classes[0]: discriminator_encoder_0,
                                      self.classes[1]: discriminator_encoder_1}
        self.discriminator_encoder = nn.ModuleDict(self.discriminator_encoder)

        self.discriminator_mlp = {self.classes[0]: MLP(input_size=opt.dis_bottleneck_size, output_size=1,
                                                       hidden_size=opt.hidden_neurons, num_blocks=3,
                                                       normalization=opt.discriminator_norm, rate=0.3,
                                                       activation=opt.discriminator_activation),
                                  self.classes[1]: MLP(input_size=opt.dis_bottleneck_size, output_size=1,
                                                       hidden_size=opt.hidden_neurons, num_blocks=3,
                                                       normalization=opt.discriminator_norm, rate=0.3,
                                                       activation=opt.discriminator_activation)}
        self.discriminator_mlp = nn.ModuleDict(self.discriminator_mlp)

        self.device = opt.device
        self.to(opt.device)

        # initialization of the weights
        self.apply(weights_init)
        if self.SVR_0:
            self.content_encoder[self.classes[0]].apply(weights_init_svr)
            self.style_encoder[self.classes[0]].apply(weights_init_svr)
        if self.SVR_1:
            self.content_encoder[self.classes[1]].apply(weights_init_svr)
            self.style_encoder[self.classes[1]].apply(weights_init_svr)

        self.eval()

    def forward(self, x, content_class, style_class, train=True):
        """
        :param x: a dictionary containing an input pair.
        :param content_class: category label for the input from which to extract the content.
        :param style_class: style label for the input from which to extract the style.
        :param train: boolean, True if training.
        :return:
        """
        # If content_class == style_class => 3D Reconstruction
        # If content_class != style_class => 3D Style Transfer

        # Extract content from the desired image x[content_class]
        content = self.content_encoder[content_class](x[content_class])
        # Extract style from the desired image x[style_class]
        style = self.style_encoder[style_class](x[style_class])
        # Decode latent codes to output pointcloud
        out = self.decoder[style_class](content, style, train=train)
        return out

    def generate_mesh(self, x, content_class, style_class):
        # If content_class == style_class => 3D Reconstruction
        # If content_class != style_class => 3D Style Transfer

        x_type_content = 'image' if x[content_class]['svr'] else 'points'
        x_type_style = 'image' if x[style_class]['svr'] else 'points'

        # Extract content from the desired image x[content_class]
        content = self.content_encoder[content_class](x[content_class][x_type_content])
        # Extract style from the desired image x[style_class]
        style = self.style_encoder[style_class](x[style_class][x_type_style])
        # Decode latent codes to output pointcloud
        out = self.decoder[style_class].generate_mesh(content, style)

        return out

    def get_latent_codes(self, x, category):
        x_type = 'image' if x[category]['svr'] else 'points'

        # Extract content from the desired image x[content_class]
        content = self.content_encoder[category](x[category][x_type])
        # Extract style from the desired image x[style_class]
        style = self.style_encoder[category](x[category][x_type])

        return content, style

    def get_latent_codes_from_sample(self, sample, category):
        # Extract content from the desired image x[content_class]
        content = self.content_encoder[category](sample['points'])
        # Extract style from the desired image x[style_class]
        style = self.style_encoder[category](sample['points'])

        return content, style

    def get_latent_content(self, x, content_class):
        x_type_content = 'image' if x[content_class]['svr'] else 'points'

        # Extract content from the desired image x[content_class]
        content = self.content_encoder[content_class](x[content_class][x_type_content])
        return content

    def get_latent_codes_with_style_sampling(self, x, content_class, style_class):
        x_type_content = 'image' if x[content_class]['svr'] else 'points'
        x_type_style = 'image' if x[style_class]['svr'] else 'points'

        # Extract content from the desired image x[content_class]
        content = self.content_encoder[content_class](x[content_class][x_type_content])
        # Extract style from the desired image x[style_class]
        style = torch.randn(x[style_class][x_type_style].size(0), self.style_bottleneck_size).to(self.device)
        style = self.style_encoder[style_class][1](style)  # passing only through the MLP

        return content, style

    def get_latent_codes_with_style_noise(self, x, content_class, style_class, noise_magnitude):
        x_type_content = 'image' if x[content_class]['svr'] else 'points'
        x_type_style = 'image' if x[style_class]['svr'] else 'points'

        # Extract content from the desired image x[content_class]
        content = self.content_encoder[content_class](x[content_class][x_type_content])
        # Extract style from the desired image x[style_class]
        style = self.style_encoder[style_class][0](x[content_class][x_type_style])
        noise = torch.randn(x[style_class][x_type_style].size(0), self.style_bottleneck_size).to(self.device)
        noise *= noise_magnitude
        style += noise
        style = self.style_encoder[style_class][1](style)  # passing only through the MLP

        return content, style

    def get_pair_of_latent_codes_with_style_noise(self, x, content_class, style_class, noise_magnitude):
        x_type_content = 'image' if x[content_class]['svr'] else 'points'
        x_type_style = 'image' if x[style_class]['svr'] else 'points'

        # Extract content from the desired image x[content_class]
        content = self.content_encoder[content_class](x[content_class][x_type_content])
        # Extract style from the desired image x[style_class]
        style_a = self.style_encoder[style_class][0](x[content_class][x_type_style])
        noise_a = torch.randn(x[style_class][x_type_style].size(0), self.style_bottleneck_size).to(self.device)
        noise_a *= noise_magnitude
        style_a += noise_a
        style_a = self.style_encoder[style_class][1](style_a)  # passing only through the MLP
        # Extract style from the desired image x[style_class]
        style_b = self.style_encoder[style_class][0](x[content_class][x_type_style])
        noise_b = torch.randn(x[style_class][x_type_style].size(0), self.style_bottleneck_size).to(self.device)
        noise_b *= noise_magnitude
        style_b += noise_b
        style_b = self.style_encoder[style_class][1](style_b)  # passing only through the MLP

        return content, style_a, style_b

    def get_pair_of_latent_styles_with_style_noise(self, x, content_class, style_class, noise_magnitude):
        x_type_content = 'image' if x[content_class]['svr'] else 'points'
        x_type_style = 'image' if x[style_class]['svr'] else 'points'

        # Extract style from the desired image x[style_class]
        style_a = self.style_encoder[style_class][0](x[content_class][x_type_style])
        noise_a = torch.randn(x[style_class][x_type_style].size(0), self.style_bottleneck_size).to(self.device)
        noise_a *= noise_magnitude
        style_a += noise_a
        style_a = self.style_encoder[style_class][1](style_a)  # passing only through the MLP
        # Extract style from the desired image x[style_class]
        style_b = self.style_encoder[style_class][0](x[content_class][x_type_style])
        noise_b = torch.randn(x[style_class][x_type_style].size(0), self.style_bottleneck_size).to(self.device)
        noise_b *= noise_magnitude
        style_b += noise_b
        style_b = self.style_encoder[style_class][1](style_b)  # passing only through the MLP

        return style_a, style_b

    def generate_mesh_from_latent_codes(self, content, style, style_class=None):
        # Decode latent codes to output mesh
        if style_class is not None:
            out = self.decoder[style_class].generate_mesh(content, style)
        return out

    def generate_pointcloud_from_latent_codes(self, content, style, style_class=None, train=False):
        """
        :param content: content latent code.
        :param style: style latent code.
        :param style_class: style label for the input from which to extract the style.
        :param train: boolean, True if training.
        :return:
        """
        # Decode latent codes to output pointcloud
        if style_class is not None:
            out = self.decoder[style_class](content, style, train=train)
        return out

    def discriminate(self, x, style_class):
        out = self.discriminator_encoder[style_class](x)
        out = self.discriminator_mlp[style_class](out)
        return out

    def fuse_primitives(self, points, faces, train=False, sample=True):
        """
        Merge generated surface elements in a single one and prepare data for Chamfer
        Input size : batch, prim, 3, npoints
        Output size : prim, prim*npoints, 3
        :return:
        """
        if self.decoder_type.lower() == "atlasnet":
            points = points.transpose(2, 3).contiguous()
            points = points.view(points.size(0), -1, 3)
        elif self.decoder_type.lower() == "meshflow" and sample and train:
            # meshes = Meshes(verts=points, faces=faces)
            # points = sample_points_from_meshes(meshes, num_samples=points.size(1))
            pass
            # TODO(msegu): sampling before the discriminator probably introduces too much noise in the discrimination
        return points.transpose(2, 1).contiguous()


class StyleNet(StyleNetBase):
    """
    Wrapper for a encoder and a decoder.
    """

    def __init__(self, opt):
        super(StyleNet, self).__init__(opt)

    def generator_update_forward(self, x, train=True):
        # Encode
        content_0 = self.content_encoder[self.classes[0]](x[self.classes[0]][self.type_0])
        content_1 = self.content_encoder[self.classes[1]](x[self.classes[1]][self.type_1])
        style_0_prime = self.style_encoder[self.classes[0]](x[self.classes[0]][self.type_0])
        style_1_prime = self.style_encoder[self.classes[1]](x[self.classes[1]][self.type_1])

        # Decode latent codes (within domain)
        out_00 = self.decoder[self.classes[0]](content_0, style_0_prime, train=train)
        fused_out_00 = self.fuse_primitives(out_00['points_3'], out_00['faces'], train, self.sample_points_from_mesh)
        out_11 = self.decoder[self.classes[1]](content_1, style_1_prime, train=train)
        fused_out_11 = self.fuse_primitives(out_11['points_3'], out_11['faces'], train, self.sample_points_from_mesh)
        # Decode latent codes (cross domain)
        out_01 = self.decoder[self.classes[1]](content_0, style_1_prime, train=train)
        fused_out_01 = self.fuse_primitives(out_01['points_3'], out_01['faces'], train, self.sample_points_from_mesh)
        out_10 = self.decoder[self.classes[0]](content_1, style_0_prime, train=train)
        fused_out_10 = self.fuse_primitives(out_10['points_3'], out_10['faces'], train,self.sample_points_from_mesh)

        # Encode again
        # Here we must take into account only the content encoder that is not doing SVR
        cycle = True
        if self.SVR_0 and self.SVR_1:
            cycle = False
        elif self.SVR_0:
            content_01 = self.content_encoder[self.classes[1]](fused_out_01)
            style_01 = self.style_encoder[self.classes[1]](fused_out_01)
            content_10 = self.content_encoder[self.classes[1]](fused_out_10)
            style_10 = None
        elif self.SVR_1:
            content_01 = self.content_encoder[self.classes[0]](fused_out_01)
            style_01 = None
            content_10 = self.content_encoder[self.classes[0]](fused_out_10)
            style_10 = self.style_encoder[self.classes[0]](fused_out_10)
        else:
            content_01 = self.content_encoder[self.classes[1]](fused_out_01)
            style_01 = self.style_encoder[self.classes[1]](fused_out_01)
            content_10 = self.content_encoder[self.classes[0]](fused_out_10)
            style_10 = self.style_encoder[self.classes[0]](fused_out_10)

        cycle = self.cycle_reconstruction and cycle
        # Decode again (combine with original style to reconstruct original pointcloud)
        if cycle:
            cycle_out_010 = self.decoder[self.classes[0]](content_01, style_0_prime, train=train)
            cycle_out_101 = self.decoder[self.classes[1]](content_10, style_1_prime, train=train)

        # Classify domain membership for each style transferred pointcloud
        class_00 = self.discriminate(fused_out_00, self.classes[0])
        class_01 = self.discriminate(fused_out_01, self.classes[1])
        class_11 = self.discriminate(fused_out_11, self.classes[1])
        class_10 = self.discriminate(fused_out_10, self.classes[0])

        out_0 = {'reconstruction': out_00['points_3'],  # this is the final reconstruction
                 'reconstruction_1': None if not 'points_1' in out_00 else out_00['points_1'],
                 'reconstruction_2': None if not 'points_2' in out_00 else out_00['points_2'],
                 'faces': None if not 'faces' in out_00 else out_00['faces'],
                 'style_transfer': out_01['points_3'],
                 'cycle_reconstruction': None if not cycle else cycle_out_010['points_3'],  # this is the final reconstruction
                 'cycle_reconstruction_1': None if (not cycle or not 'points_1' in cycle_out_010) else cycle_out_010['points_1'],
                 'cycle_reconstruction_2': None if (not cycle or not 'points_2' in cycle_out_010) else cycle_out_010['points_2'],
                 'reconstruction_logits': class_00,
                 'style_transfer_logits': class_01,
                 'content_code': content_0,
                 'style_code': style_0_prime,
                 'cycle_content_code': content_01,
                 'cycle_style_code': style_10}  # cycle_style_code: E^s_0((D_0(E^c(x_1), E^s_0(x_0)))

        out_1 = {'reconstruction': out_11['points_3'], # this is the final reconstruction
                 'reconstruction_1': None if not 'points_1' in out_11 else out_11['points_1'],
                 'reconstruction_2': None if not 'points_2' in out_11 else out_11['points_2'],
                 'faces': None if not 'faces' in out_11 else out_11['faces'],
                 'style_transfer': out_10['points_3'],
                 'cycle_reconstruction': None if not cycle else cycle_out_101['points_3'],  # this is the final reconstruction
                 'cycle_reconstruction_1': None if (not cycle or not 'points_1' in cycle_out_101) else cycle_out_101['points_1'],
                 'cycle_reconstruction_2': None if (not cycle or not 'points_2' in cycle_out_101) else cycle_out_101['points_2'],
                 'reconstruction_logits': class_11,
                 'style_transfer_logits': class_10,
                 'content_code': content_1,
                 'style_code': style_1_prime,
                 'cycle_content_code': content_10,
                 'cycle_style_code': style_01}  # cycle_style_code: E^s_1((D_1(E^c(x_0), E^s_1(x_1)))

        return out_0, out_1

    def discriminator_update_forward(self, x, train=True):
        # Encode
        content_0 = self.content_encoder[self.classes[0]](x[self.classes[0]][self.type_0])
        content_1 = self.content_encoder[self.classes[1]](x[self.classes[1]][self.type_1])
        style_0_prime = self.style_encoder[self.classes[0]](x[self.classes[0]][self.type_0])
        style_1_prime = self.style_encoder[self.classes[1]](x[self.classes[1]][self.type_1])

        # Decode latent codes to output pointcloud
        out_01 = self.decoder[self.classes[1]](content_0, style_1_prime, train=train)
        fused_out_01 = self.fuse_primitives(out_01['points_3'], out_01['faces'], train, self.sample_points_from_mesh)
        # Decode latent codes to output pointcloud
        out_10 = self.decoder[self.classes[0]](content_1, style_0_prime, train=train)
        fused_out_10 = self.fuse_primitives(out_10['points_3'], out_10['faces'], train, self.sample_points_from_mesh)

        # Classify domain membership for each style transferred pointcloud
        class_0 = self.discriminate(x[self.classes[0]]['points'], self.classes[0])
        class_1 = self.discriminate(x[self.classes[1]]['points'], self.classes[1])
        class_01 = self.discriminate(fused_out_01, self.classes[1])
        class_10 = self.discriminate(fused_out_10, self.classes[0])

        out_0 = {'style_transfer': out_01['points_3'],
                 'faces': None if not 'faces' in out_01 else out_01['faces'],
                 'reconstruction_logits': class_0,
                 'style_transfer_logits': class_10}

        out_1 = {'style_transfer': out_10['points_3'],
                 'faces': None if not 'faces' in out_10 else out_10['faces'],
                 'reconstruction_logits': class_1,
                 'style_transfer_logits': class_01}

        return out_0, out_1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if classname.find('Adaptive') != -1:
            m.norm.weight.data.normal_(1.0, 0.02)
            m.norm.bias.data.fill_(0)
        else:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def weights_init_svr(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if classname.find('Adaptive') != -1:
            nn.init.ones_(m.norm.weight)
            nn.init.zeros_(m.norm.bias)
        else:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


