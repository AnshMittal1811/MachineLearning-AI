"""
Copyright (c) 2021, Mattia Segu
Licensed under the MIT License (see LICENSE for details)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from auxiliary.ChamferDistancePytorch.fscore import fscore
import os



class TrainerLoss(object):
    """
    This class implements all functions related to the loss of 3DSNet.
    """

    def __init__(self, opt):
        super(TrainerLoss, self).__init__(opt)
        # Losses weights
        self.weight_chamfer = opt.weight_chamfer
        self.weight_cycle_chamfer = opt.weight_cycle_chamfer
        self.weight_adversarial = opt.weight_adversarial
        self.weight_perceptual = opt.weight_perceptual
        self.weight_content_reconstruction = opt.weight_content_reconstruction
        self.weight_style_reconstruction = opt.weight_style_reconstruction

        self.cycle_reconstruction = True if opt.weight_cycle_chamfer > 0 else False
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()
        self.gan_type = opt.gan_type

        self.weight_chamfer_1 = self.opt.w_multiscale_1
        self.weight_chamfer_2 = self.opt.w_multiscale_2
        self.weight_chamfer_3 = self.opt.w_multiscale_3

        self.instancenorm = nn.InstanceNorm1d(1024, affine=False)

    def build_losses(self):
        """
        Creates loss functions.
        """
        self.distChamfer = dist_chamfer_3D.chamfer_3DDist()
        self.generator_loss_model = self.total_generator_loss
        self.discriminator_loss_model = self.total_discriminator_loss

    def fuse_primitives(self, points, faces, sample=True):
        """
        Merge generated surface elements in a single one and prepare data for Chamfer

        :param points: input points for each existing patch to be merged
        Input size : batch, prim, 3, npoints
        Output size : prim, prim*npoints, 3

        :param faces: faces connecting points. Necessary for meshflow which needs mesh representations

        :return points: merged surface elements in a single one
        """
        if self.opt.decoder_type.lower() == "atlasnet":
            # import pdb; pdb.set_trace()
            points = points.transpose(2, 3).contiguous()
            points = points.view(points.size(0), -1, 3)
        elif self.opt.decoder_type.lower() == "meshflow" and sample and self.flags.train:
            meshes = Meshes(verts=points, faces=faces)
            points = sample_points_from_meshes(meshes, num_samples=points.size(1))
        return points

    def l1_distance(self, inputs, targets):
        return torch.mean(torch.abs(inputs - targets))

    def latent_reconstruction_loss(self, data, content_weight, style_weight):
        """
        Computes reconstruction loss in latent space as l1 distance between latent codes.

        :param data: data dictionary for the current batch of a given domain
        :param content_weight: weight to give to latent content reconstruction loss
        :param style_weight: weight to give to latent style reconstruction loss
        """
        if data.cycle_content_code is not None:
            loss = content_weight*self.l1_distance(data.content_code, data.cycle_content_code)
        if data.cycle_style_code is not None:
            loss += style_weight*self.l1_distance(data.style_code, data.cycle_style_code)
        data.loss += loss

    def perceptual_distance(self, source, target):
        """
        Computes perceptual distance.

        :param source: generated 3D object
        :param target: ground truth 3D object to compute distance from
        """
        if self.opt.perceptual_by_layer and (self.flags.train):
        # if self.opt.perceptual_by_layer and (self.flags.train or self.opt.run_single_eval):
            source_features = self.perceptual_network.module.return_by_layer(source)
            target_features = self.perceptual_network.module.return_by_layer(target)
            distance = 0.0
            for source_f, target_f in zip(source_features, target_features):
                distance += torch.mean((source_f - target_f) ** 2)   # Problem with sizes used in eval!
            return distance
        else:
            source_features = self.perceptual_network(source)
            target_features = self.perceptual_network(target)
            return torch.mean((source_features - target_features) ** 2)

    def perceptual_loss(self, data_a, data_b, weight):
        """
        Computes perceptual loss.

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        """
        def make_network_input(points):
            return points.transpose(2, 1).contiguous()
        in_0 = make_network_input(data_a.points).to(self.opt.device)
        in_1 = make_network_input(data_b.points).to(self.opt.device)
        out_00 = make_network_input(data_a.reconstruction).to(self.opt.device)
        out_01 = make_network_input(data_a.style_transfer).to(self.opt.device)
        out_11 = make_network_input(data_b.reconstruction).to(self.opt.device)
        out_10 = make_network_input(data_b.style_transfer).to(self.opt.device)
        if out_00.size() != in_0.size():
            pass

        # Distances for loss_0
        d_00_0 = self.perceptual_distance(out_00, in_0)
        d_00_1 = self.perceptual_distance(out_00, in_1)
        d_01_1 = self.perceptual_distance(out_01, in_1)
        d_01_0 = self.perceptual_distance(out_01, in_0)
        # Distances for loss_1
        d_11_1 = self.perceptual_distance(out_11, in_1)
        d_11_0 = self.perceptual_distance(out_11, in_0)
        d_10_0 = self.perceptual_distance(out_10, in_0)
        d_10_1 = self.perceptual_distance(out_10, in_1)
        # Loss
        data_a.loss += d_01_1*weight
        data_b.loss += d_10_0*weight

        data_a.lpips_rec_from_source, data_b.lpips_rec_from_source = d_00_0, d_11_1
        data_a.lpips_rec_from_target, data_b.lpips_rec_from_target = d_00_1, d_11_0
        data_a.lpips_from_source, data_b.lpips_from_source = d_01_0, d_10_1
        data_a.lpips_from_target, data_b.lpips_from_target = d_01_1, d_10_0

    def chamfer_loss(self, data, weight):
        """
        Component of the training loss of 3DSNet. The Chamfer Distance. Compute the f-score in eval mode.

        :param data: data dictionary for the current batch of a given domain
        :param weight: weight to assign to this loss component
        """
        inCham1 = data.points.view(data.points.size(0), -1, 3).contiguous()
        inCham2 = data.reconstruction.contiguous().view(data.points.size(0), -1, 3).contiguous()

        dist1, dist2, idx1, idx2 = self.distChamfer(inCham1, inCham2)  # mean over points
        data.loss += weight*(torch.mean(dist1) + torch.mean(dist2))  # mean over points
        if not self.flags.train:
            data.loss_fscore, _, _ = fscore(dist1, dist2)
            data.loss_fscore = data.loss_fscore.mean()
            data.chamfer_distance = torch.mean(dist1) + torch.mean(dist2)


    def multioutput_chamfer_loss(self, data, weight):
        """
        Component of the training loss of 3DSNet. The Multioutput Chamfer Distance. Computes the f-score in eval mode.
        Used when the generated 3D object is given at different levels of the network.
        In particular, it is used with our Adaptive-Meshflow backbone.

        :param data: data dictionary for the current batch of a given domain
        :param weight: weight to assign to this loss component
        """
        inCham0 = data.points.view(data.points.size(0), -1, 3).contiguous()
        inCham1 = data.reconstruction_1.contiguous().view(data.points.size(0), -1, 3).contiguous()
        inCham2 = data.reconstruction_2.contiguous().view(data.points.size(0), -1, 3).contiguous()
        inCham3 = data.reconstruction.contiguous().view(data.points.size(0), -1, 3).contiguous()

        # After first transformation block
        dist1, dist2, idx1, idx2 = self.distChamfer(inCham0, inCham1)  # mean over points
        data.loss += weight * self.weight_chamfer_1 * (torch.mean(dist1) + torch.mean(dist2))  # mean over points

        # After second transformation block
        dist1, dist2, idx1, idx2 = self.distChamfer(inCham0, inCham2)  # mean over points
        data.loss += weight * self.weight_chamfer_2 * (torch.mean(dist1) + torch.mean(dist2))  # mean over points

        # After third transformation block
        dist1, dist2, idx1, idx2 = self.distChamfer(inCham0, inCham3)  # mean over points
        data.loss += weight * self.weight_chamfer_3 *(torch.mean(dist1) + torch.mean(dist2))  # mean over points
        if not self.flags.train:
            data.loss_fscore, _, _ = fscore(dist1, dist2)
            data.loss_fscore = data.loss_fscore.mean()
            data.chamfer_distance = torch.mean(dist1) + torch.mean(dist2)

    def cycle_chamfer_loss(self, data, weight):
        """
        Component of the training loss of 3DSNet. The Cycle Chamfer Distance. Compute the cycle f-score in eval mode.

        :param data: data dictionary for the current batch of a given domain
        :param weight: weight to assign to this loss component
        """
        inCham1 = data.points.view(data.points.size(0), -1, 3).contiguous()
        inCham2 = data.cycle_reconstruction.contiguous().view(data.points.size(0), -1, 3).contiguous()

        dist1, dist2, idx1, idx2 = self.distChamfer(inCham1, inCham2)  # mean over points
        data.loss += weight*(torch.mean(dist1) + torch.mean(dist2))  # mean over points
        if not self.flags.train:
            data.loss_cycle_fscore, _, _ = fscore(dist1, dist2)
            data.loss_cycle_fscore = data.loss_cycle_fscore.mean()
            data.cycle_chamfer_distance = torch.mean(dist1) + torch.mean(dist2)

    def multioutput_cycle_chamfer_loss(self, data, weight):
        """
        Component of the training loss of 3DSNet. The Multioutput Cycle Chamfer Distance. Computes the cycle f-score in
        eval mode.
        Used when the generated 3D object is given at different levels of the network.
        In particular, it is used with our Adaptive-Meshflow backbone.

        :param data: data dictionary for the current batch of a given domain
        :param weight: weight to assign to this loss component
        """
        inCham0 = data.points.view(data.points.size(0), -1, 3).contiguous()
        inCham1 = data.cycle_reconstruction_1.contiguous().view(data.points.size(0), -1, 3).contiguous()
        inCham2 = data.cycle_reconstruction_2.contiguous().view(data.points.size(0), -1, 3).contiguous()
        inCham3 = data.cycle_reconstruction.contiguous().view(data.points.size(0), -1, 3).contiguous()

        # After first transformation block
        dist1, dist2, idx1, idx2 = self.distChamfer(inCham0, inCham1)  # mean over points
        data.loss += weight * self.weight_chamfer_1 * (torch.mean(dist1) + torch.mean(dist2))  # mean over points

        # After second transformation block
        dist1, dist2, idx1, idx2 = self.distChamfer(inCham0, inCham2)  # mean over points
        data.loss += weight * self.weight_chamfer_2 * (torch.mean(dist1) + torch.mean(dist2))  # mean over points

        # After third transformation block
        dist1, dist2, idx1, idx2 = self.distChamfer(inCham0, inCham3)  # mean over points
        data.loss += weight * self.weight_chamfer_3 *(torch.mean(dist1) + torch.mean(dist2))  # mean over points
        if not self.flags.train:
            data.loss_cycle_fscore, _, _ = fscore(dist1, dist2)
            data.loss_cycle_fscore = data.loss_fscore.mean()
            data.cycle_chamfer_distance = torch.mean(dist1) + torch.mean(dist2)

    def discriminator_adversarial_loss(self, data, weight):
        """
        Component of the training loss of 3DSNet. Adversarial loss as used during discriminator update step.

        :param data: data dictionary for the current batch of a given domain
        :param weight: weight to assign to this loss component
        """
        if self.gan_type == 'lsgan':
            adversarial_loss = torch.mean((data.style_transfer_logits-0)**2)
            adversarial_loss += torch.mean((data.reconstruction_logits - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all_zeros = torch.zeros_like(data.style_transfer_logits).to(self.opt.device)
            all_ones = torch.ones_like(data.reconstruction_logits).to(self.opt.device)
            adversarial_loss = self.binary_cross_entropy(data.style_transfer_logits, all_zeros)
            adversarial_loss += self.binary_cross_entropy(data.reconstruction_logits, all_ones)
        else:
            assert 0, 'Unsupported GAN type: {}'.format(self.gan_type)
        data.loss += weight*adversarial_loss

    def generator_adversarial_loss(self, data, weight):
        """
        Component of the training loss of 3DSNet. Adversarial loss as used during generator update step.

        :param data: data dictionary for the current batch of a given domain
        :param weight: weight to assign to this loss component
        """
        if self.gan_type == 'lsgan':
            adversarial_loss = torch.mean((data.style_transfer_logits-1)**2)
        elif self.gan_type == 'nsgan':
            all_ones = torch.ones_like(data.style_transfer_logits).to(self.opt.device)
            adversarial_loss = self.binary_cross_entropy(data.style_transfer_logits, all_ones)
        else:
            assert 0, 'Unsupported GAN type: {}'.format(self.gan_type)

        data.loss += weight*adversarial_loss

    def total_discriminator_loss(self, data_a, data_b):
        """
        Calls all loss models for discriminator training. Generator is fixed.

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        """
        # Adversarial loss
        self.discriminator_adversarial_loss(data_a, self.opt.weight_adversarial)
        self.discriminator_adversarial_loss(data_b, self.opt.weight_adversarial)

    def total_generator_loss(self, data_a, data_b):
        """
        Calls all loss models for generator training. Discriminator is fixed.

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        :return:
        """
        # We follow MUNIT employing bidirectional reconstruction loss:
        # Image reconstruction loss
        if self.opt.decoder_type.lower() == 'meshflow' and self.opt.multiscale_loss:
            self.multioutput_chamfer_loss(data_a, self.weight_chamfer)
            self.multioutput_chamfer_loss(data_b, self.weight_chamfer)
        else:
            self.chamfer_loss(data_a, self.weight_chamfer)
            self.chamfer_loss(data_b, self.weight_chamfer)

        # Latent reconstruction loss
        self.latent_reconstruction_loss(data_a,
                                        self.weight_content_reconstruction,
                                        self.weight_style_reconstruction)
        self.latent_reconstruction_loss(data_b,
                                        self.weight_content_reconstruction,
                                        self.weight_style_reconstruction)
        # Cycle reconstruction loss
        if self.cycle_reconstruction:
            if self.opt.decoder_type.lower() == 'meshflow' and self.opt.multiscale_loss:
                self.multioutput_cycle_chamfer_loss(data_a, self.weight_cycle_chamfer)
                self.multioutput_cycle_chamfer_loss(data_b, self.weight_cycle_chamfer)
            else:
                self.cycle_chamfer_loss(data_a, self.weight_cycle_chamfer)
                self.cycle_chamfer_loss(data_b, self.weight_cycle_chamfer)

        # Adversarial loss
        self.generator_adversarial_loss(data_a, self.weight_adversarial)
        self.generator_adversarial_loss(data_b, self.weight_adversarial)

        # Perceptual loss
        if os.path.exists(self.opt.reload_pointnet_path):
            self.perceptual_loss(data_a, data_b, self.weight_perceptual)

    def metro(self):
        """
        Compute the metro distance on a randomly selected test files.
        Uses joblib to leverage as much cpu as possible
        :return:
        """
        # TODO(msegu): if you want you can compute the metro distance on some randomly selected files in SMAL
        #              See implementation in AtlasNet official repository
        pass

