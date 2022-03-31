import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

import modules
import modules_unet

import sdf_rendering
import utils.common_utils as common_utils
import utils.activations as activations
from utils.ray_builder import RayBuilder
import utils.diff_operators as diff_operators
import utils.math_utils_torch as mut
import numbers
import time
from scipy.spatial.transform import Rotation as R
import os


class SDFIBRNet(MetaModule):
    """
    Master network for both sdf and image feature extractors.
    """

    def __init__(self, opt, ray_builder):
        super().__init__()
        self.opt = opt
        if type(ray_builder) == list:
            self.opt.is_meta = 1
        else:
            self.opt.is_meta = 0
        self.ray_builder = ray_builder

        # Model.
        activation = [opt.model_activation_sdf]
        activation_last = ['none']
        num_hidden_layers = [opt.model_hidden_layers_sdf]
        num_hidden_features = [opt.model_hidden_dims_sdf]

        # Image encoding, decoding, and aggregation models
        self.enc_net = modules_unet.ResUNet_Meta(out_channels_0=opt.model_image_encoder_features,
                                                 depth=opt.model_image_encoder_depth)

        # Use standard ResNet
        self.dec_net = modules_unet.UNet_Meta(in_channels=opt.model_image_encoder_features, opt=opt)

        # Skip connections.
        model_skips_sdf = np.array(common_utils.parse_comma_int_args(opt.model_skips_sdf)).reshape(-1, 2)
        skip_connections = [model_skips_sdf]

        # Input/Outputs.
        in_features = [3]  # pos
        out_features = [1]  # sdf

        # For loading in model with more features output from sdf
        if opt.feature_vector > 0:
            out_features[0] += opt.feature_vector

        # Positional encoding.
        positional_encoding = [opt.posenc_sdf]
        positional_encoding_kwargs = [
            {
                'num_bands': opt.posenc_sdf_bands,
                'bands_exp': opt.posenc_sdf_expbands,
                'channels': None,
                'sigma': opt.posenc_sdf_sigma,
            },
        ]

        self.out_features = out_features
        self.in_features = in_features

        # Enable flow MLP for video.
        decoders_cls = [modules.SDFDecoder]
        self.decoders = [decoders_cls[i](opt, in_features=in_features[i],
                                         hidden_features=num_hidden_features[i],
                                         out_features=out_features[i],
                                         num_hidden_layers=num_hidden_layers[i],
                                         skip_connections=skip_connections[i],
                                         activation=activation[i],
                                         activation_last=activation_last[i],
                                         positional_encoding=positional_encoding[i],
                                         positional_encoding_kwargs=positional_encoding_kwargs[i])
                         for i in range(len(decoders_cls))]
        self.decoder_sdf = self.decoders[0]

        if opt.init_sphere:
            print('Initializing sphere SDF...')
            if activation[0] in ['relu', 'softplus', 'swish']:
                # Analytical initialization (only for relu).
                activations.sdf_sphere_init_full(self.decoder_sdf.net)
            else:
                # Numerical initialization.
                filename = f'fit_sphere_{activation[0]}'
                if self.out_features[0] > 1:
                    filename += f'_{self.out_features[0]}'

                if self.opt.dataset_name == 'dtu' or self.opt.dataset_name == 'shapenet':
                    filename = 'sphere_sine128x5'
                elif self.opt.dataset_name == 'nlr':
                    filename = 'sphere_sine256x5'
                folder = 'regularized' if self.opt.init_regularized else 'base'
                self.load_checkpoint(f'./assets/{folder}/{filename}.pth', True, False)

        if opt.init_plane and opt.model == 'ours' and activation[0] == 'sine':
            print('Initializing plane SDF...')
            # Numerical initialization.
            self.load_checkpoint('./assets/siren_plane.pth', True, False)

        # Get feature aggregation method, and feature types.
        if opt.feature_aggregation_method == "sum":
            self.aggregation_fn = self.aggregate_features_sum
        elif opt.feature_aggregation_method == "mean":
            self.aggregation_fn = self.aggregate_features_mean
        elif opt.feature_aggregation_method == "lumigraph":
            self.aggregation_fn = self.aggregate_features_lumigraph
        elif opt.feature_aggregation_method == "lumigraph_epipolar":
            self.aggregation_fn = self.aggregate_features_lumigraph_epipolar
        elif opt.feature_aggregation_method == "learned":
            self.aggregation_fn = self.aggregate_features_learned

            self.agg_net = modules.FCBlock(in_features=opt.model_image_encoder_features + 3,
                                           # out_features=opt.model_image_encoder_features + 1,
                                           out_features=1,
                                           num_hidden_layers=3, hidden_features=32, outermost_linear=False,
                                           activation_last='sigmoid')
        else:
            raise RuntimeError("Not implemented aggregation function.")

        if opt.feature_type == "learned":
            self.encoding_fn = self.encode_images
            self.decoding_fn = self.decode_image
        elif opt.feature_type == "rgb":
            self.encoding_fn = self.encode_images_rgb
            self.decoding_fn = self.decode_image_rgb
        else:
            raise RuntimeError("Invalid feature type.")

        if opt.occlusion_method == "correlation":
            self.occlusion_fn = self.get_correlation_mask
        else:
            self.occlusion_fn = self.get_occlusion_mask

        # Get total number of views / views per target
        if opt.is_meta:
            self.total_number_views = [len(rb['ray_builder'].img_dataset.frames[0].image_views) for rb in self.ray_builder]
        else:
            self.total_number_views = len(self.ray_builder.img_dataset.frames[0].image_views)
        if self.opt.total_number_source_views > 0:
            if opt.is_meta:
                self.total_number_views = [self.opt.total_number_source_views for _ in range(len(self.ray_builder))]
            else:
                self.total_number_views = self.opt.total_number_source_views

        if self.opt.source_views_per_target == -1:
            if opt.is_meta:
                self.opt.source_views_per_target = [tv-1-len(opt.WITHHELD_VIEWS) for tv in self.total_number_views]
            else:
                self.opt.source_views_per_target = self.total_number_views - 1 - len(opt.WITHHELD_VIEWS)
        elif opt.is_meta:
            self.opt.source_views_per_target = [self.opt.source_views_per_target for _ in range(len(self.total_number_views))]

        if opt.source_view_selection_mode == "random":
            self.source_view_select_fn = self.source_view_select_random_withheld
        elif opt.source_view_selection_mode == "nearest":
            self.source_view_select_fn = self.source_view_select_nearest
            self.extrinsic_similarity_matrix_init = self.compute_extrinsic_similarity()
        else:
            raise RuntimeError("Invalid selection type")

        self.WITHHELD_VIEWS = opt.WITHHELD_VIEWS

    def re_init_sphere(self):
        if self.opt.dataset_name == 'dtu' or self.opt.dataset_name == 'shapenet':
            filename = 'sphere_sine128x5'
        elif self.opt.dataset_name == 'nlr':
            filename = 'sphere_sine256x5'
        folder = 'regularized' if self.opt.init_regularized else 'base'
        self.load_checkpoint(f'./assets/{folder}/{filename}.pth', load_sdf=True)

    def re_init_agg(self):
        self.agg_net = modules.FCBlock(in_features=self.opt.model_image_encoder_features + 3,
                                       # out_features=opt.model_image_encoder_features + 1,
                                       out_features=1,
                                       num_hidden_layers=3, hidden_features=32, outermost_linear=False,
                                       activation_last='sigmoid').to(self.device)

    @torch.no_grad()
    def compute_extrinsic_similarity(self):
        """
        Converts list of extrinsics to similarity matrix using inner product of extrinsic
        values. Could be done in a better way, but this approximately works.
        """
        if self.opt.is_meta:
            dataset_sim_matrices = []
            for i, dataset_total_views in enumerate(self.total_number_views):
                extrinsics = []
                for view_id in range(dataset_total_views):
                    extrinsic = self.ray_builder[i]['ray_builder'].img_dataset.frames[0].image_views[view_id].extrinsics
                    extrinsics += [extrinsic]

                extrinsics = torch.stack(extrinsics, 0)
                sim_matrix = torch.matmul(extrinsics, extrinsics.transpose(0, 1))
                dataset_sim_matrices.append(sim_matrix)
            return dataset_sim_matrices
        else:
            extrinsics = []
            for view_id in range(self.total_number_views):
                extrinsic = self.ray_builder.img_dataset.frames[0].image_views[view_id].extrinsics
                extrinsics += [extrinsic]

            extrinsics = torch.stack(extrinsics, 0)
            sim_matrix = torch.matmul(extrinsics, extrinsics.transpose(0, 1))
            return sim_matrix

    @torch.no_grad()
    def precompute_3D_buffers(self, dataset_num=None, num_precompute=None):
        """
        Precompute 3D position buffers for all images using the loaded SDF model. No gradient computation.
        """
        ray_builder_curr = self.get_ray_builder(dataset_num)
        if self.opt.is_meta:
            source_view_ids = list(range(len(ray_builder_curr.img_dataset.frames[0].image_views)))
            if dataset_num is not None:
                source_view_ids = source_view_ids[:self.total_number_views[dataset_num]]
        else:
            source_view_ids = self.opt.TRAIN_VIEWS + self.opt.WITHHELD_VIEWS
        self.precomputed_buffers = self.compute_3D_buffers(source_view_ids, grad=0, dataset_num=dataset_num)

    @torch.no_grad()
    def load_3D_buffers(self, filepath):
        self.precomputed_buffers = torch.load(filepath)
        assert(len(self.precomputed_buffers) == len(self.ray_builder.img_dataset.frames[0].image_views))

    @torch.no_grad()
    def save_3D_buffers(self, filepath):
        torch.save(self.precomputed_buffers, filepath)

    def get_ray_builder(self, dataset_num):
        if dataset_num is not None and self.opt.is_meta:
            ray_builder_curr = self.ray_builder[dataset_num]['ray_builder']
        else:
            ray_builder_curr = self.ray_builder
        return ray_builder_curr

    def compute_3D_buffers(self, source_view_ids, grad=0, params=None, dataset_num=None):
        """
        Compute the 3D position buffers for all source views using the current SDF model.
        """
        ray_builder_curr = self.get_ray_builder(dataset_num)
        precomputed_buffers = {}

        model_matrix = ray_builder_curr.model_matrix
        for view_id in source_view_ids:
            view = ray_builder_curr.img_dataset.frames[0].image_views[view_id]
            proj_matrix = view.projection_matrix
            view_matrix = view.view_matrix
            resolution = torch.from_numpy(view.resolution).to(self.device)

            # Only need gradient for computation of target view. Rest are used for finding masks.
            # if grad and view_id == source_view_ids[-1]:
            if grad:
                res = sdf_rendering.render_view_proj_differentiable(self, resolution, model_matrix, view_matrix,
                                                                    proj_matrix, -1, batch_size=0, params=params)
                precomputed_buffers[view_id] = res
            else:
                res = sdf_rendering.render_view_proj(self, resolution, model_matrix, view_matrix,
                                                     proj_matrix, -1, batch_size=0, params=params)
                precomputed_buffers[view_id] = {'pos': res['pos']}

        return precomputed_buffers

    def project_view(self, source_pos, model_matrix, target_view_matrix, target_proj_matrix):
        """
        Project source 3D positions to another view.
        """
        # Convert to homogeneous coordinates, 4 x N.
        resolution = source_pos.shape[0:2]
        source_pos = source_pos.view(-1, 3).transpose(0, 1)
        source_pos_homog = torch.cat((source_pos, torch.ones_like(source_pos)[0].unsqueeze(0)), 0)

        # Project 3D source points to other viewpoint.
        projected_pos_homog = target_proj_matrix @ target_view_matrix @ model_matrix @ source_pos_homog

        # Convert into in projected space x/w, y/w, z/w. Clamp to 1e-8 to prevent NANs in backward pass?
        # projected_pos = projected_pos_homog[:3, :] / projected_pos_homog[3, :].unsqueeze(0)
        # projected_pos[projected_pos < 1e-8] = 0
        projected_pos = projected_pos_homog[:3, :] / torch.clamp(projected_pos_homog[3, :].unsqueeze(0), min=1e-8)
        projected_pos = projected_pos.transpose(0, 1).reshape(resolution[0], resolution[1], 3)

        return projected_pos

    def project_target_view_to_sources(self, target_view_id_or_trgt_res,
                                       source_view_ids, computed_buffers=None, dataset_num=None):
        """
        Project target view 3D positions into all of the source views.
        """
        ray_builder_curr = self.get_ray_builder(dataset_num)

        all_projected_pos = []
        model_matrix = ray_builder_curr.model_matrix
        if computed_buffers is None:
            trgt_res = target_view_id_or_trgt_res
        else:
            trgt_res = computed_buffers[target_view_id_or_trgt_res]

        # Iterate through all source views.
        for i, view_id in enumerate(source_view_ids):
            view = ray_builder_curr.img_dataset.frames[0].image_views[view_id]
            src_proj_matrix = view.projection_matrix
            src_view_matrix = view.view_matrix
            projected_pos = self.project_view(trgt_res['pos'], model_matrix, src_view_matrix, src_proj_matrix)

            all_projected_pos += [projected_pos]

        # Stack points in all other views. No check if in frustum of camera, or occlusion.
        all_projected_pos = torch.stack(all_projected_pos, 0)
        return all_projected_pos

    @torch.no_grad()
    def get_frustum_mask(self, projected_pos_maps):
        """
        Mask out non-visible points in camera frustum of projected positions. Return this mask.
        """
        frustum_mask = torch.zeros_like(projected_pos_maps)[..., 0].bool()

        for i in range(projected_pos_maps.shape[0]):
            projected_pos = projected_pos_maps[i]

            cube_check = torch.logical_and(projected_pos <= 1, projected_pos >= -1)
            xy_check = torch.logical_and(cube_check[..., 0], cube_check[..., 1])

            frustum_mask[i] = xy_check

        return frustum_mask

    @torch.no_grad()
    def get_occlusion_mask(self, projected_pos_maps, frustum_mask, target_view_id_or_res, source_view_ids,
                           computed_buffers, image_features=None, dataset_num=None):
        """
        Mask out occluded target points in each of the source views. Return this mask, along with the 3D points which
        actually come from these source views.
        """
        ray_builder_curr = self.get_ray_builder(dataset_num)

        if isinstance(target_view_id_or_res, int):
            trgt_res = computed_buffers[target_view_id_or_res]
        else:
            trgt_res = target_view_id_or_res

        occlusion_masks = torch.zeros_like(frustum_mask)
        sampled_3D_pos_all = torch.zeros_like(projected_pos_maps)
        occlusion_differences = torch.zeros_like(occlusion_masks)

        for i, view_id in enumerate(source_view_ids):
            source_resolution = torch.Tensor(ray_builder_curr.img_dataset.frames[0].image_views[view_id].resolution).int()

            # N x 3, 3D coordinates projected onto the source view which are in view of the frustrum. Convert from ndc to [-1,1]^2 sampling coords.
            # Reshape to form which can be used by grid_sample, 1 x 1 x N x 2.
            projected_pos_valid = projected_pos_maps[i] * frustum_mask[i].float().unsqueeze(-1)
            projected_pos_valid_xy = sdf_rendering.ndc_to_standard(projected_pos_valid[..., :2].view(-1, 2), source_resolution)
            projected_pos_valid_xy_res = projected_pos_valid_xy.view(1, 1, -1, 2)

            # Reshape positions from H x W x 3 to 1 x 3 x H x W to use the grid_sample function.
            positions_3D = computed_buffers[view_id]['pos'].permute(2, 0, 1).unsqueeze(0)

            # Sample the 3D positions using the grid_sample function. This returns output of size N x 3.
            sampled_3D_pos = F.grid_sample(positions_3D, projected_pos_valid_xy_res, align_corners=True).reshape(3, -1).transpose(0, 1)
            sampled_3D_pos_all[i] = sampled_3D_pos.view(sampled_3D_pos_all.shape[1], sampled_3D_pos_all.shape[2], 3)

            # Get 3D positions from target view.
            original_3D_pos = trgt_res['pos'].view(-1, 3)

            # Compare if each point is close in R3.
            diff = torch.sum((original_3D_pos - sampled_3D_pos) ** 2, -1)
            occlusion_mask = (diff < getattr(self.opt, "occ_threshold", 1e-5))

            occlusion_masks[i] = occlusion_mask.view(source_resolution[1], source_resolution[0])
            occlusion_differences[i] = diff.view(source_resolution[1], source_resolution[0])

        return occlusion_masks, occlusion_differences

    @torch.no_grad()
    def get_correlation_mask(self, projected_pos_maps, frustum_mask, target_view_id, source_view_ids,
                             computed_buffers, image_features=None):
        """
        Mask out occluded target points in each of the source views. Do this using correlation between the features
        of the source images and the ground truth images. Return the mask.
        """
        target_image_features = image_features[-1]

        # Compute correlation as outlined in literature (ex: Group-wise Correlation Stereo Network, eqn 3)
        image_feature_correlation = (image_features * target_image_features.unsqueeze(0)).mean(dim=1)

        # Mask out negatively correlated features, and then compute soft occlusion weights using correlation.
        corr_mask = (image_feature_correlation > 0).float() * image_feature_correlation
        occlusion_weights = image_features.shape[1] * corr_mask / corr_mask.sum(dim=0, keepdim=True)

        return occlusion_weights, image_feature_correlation

    @torch.no_grad()
    def get_rays_mask(self, projected_pos_maps, frustum_mask, target_view_id, source_view_ids, dataset_num=None):
        """
        Return the ray mask in each of the source views, along with the mask corresponding the the
        projected points from the target views. Return these two masks.
        """
        ray_builder_curr = self.get_ray_builder(dataset_num)

        rays_mask = torch.zeros_like(frustum_mask)
        proj_rays_mask = torch.zeros_like(frustum_mask)

        for i, view_id in enumerate(source_view_ids):
            view = ray_builder_curr.img_dataset.frames[0].image_views[view_id]
            source_resolution = torch.Tensor(view.resolution).int()
            rays_mask[i] = view.mask[0]

            projected_pos_valid = projected_pos_maps[i] * frustum_mask[i].float().unsqueeze(-1)
            projected_pos_valid_xy_res = sdf_rendering.ndc_to_standard(projected_pos_valid[..., :2].view(-1, 2), source_resolution).view(1,1,-1,2)

            sampled_rays_mask = F.grid_sample(rays_mask[i].unsqueeze(0).unsqueeze(0).float(),
                                              projected_pos_valid_xy_res, align_corners=True, mode='nearest').reshape(view.mask.shape)
            proj_rays_mask[i] = sampled_rays_mask[0]

        return rays_mask, proj_rays_mask

    def encode_images(self, source_view_ids, dataset_num=None, params=None):
        """
        Encode all images besides target_view_id into feature grids.
        """
        ray_builder_curr = self.get_ray_builder(dataset_num)

        images = []
        masks = []
        for i, view_id in enumerate(source_view_ids):
            images.append(ray_builder_curr.img_dataset.frames[0].image_views[view_id].image)
            masks.append(ray_builder_curr.img_dataset.frames[0].image_views[view_id].mask)

        # Stack images, normalize in [-1,1], and then encode into features.
        images = torch.stack(images, 0).to(self.device)
        images_res = (images * 2) - 1

        return self.enc_net.forward(images_res, params=get_subdict(params, 'enc_net')), images

    def encode_images_rgb(self, source_view_ids, dataset_num=None, params=None):
        ray_builder_curr = self.get_ray_builder(dataset_num)

        images = []
        for i, view_id in enumerate(source_view_ids):
            images.append(ray_builder_curr.img_dataset.frames[0].image_views[view_id].image)

        # Stack images, normalize in [-1,1].
        images = torch.stack(images, 0).to(self.device)
        images_res = (images * 2) - 1

        return images_res, images

    def sample_features(self, source_image_features, frustum_mask, occlusion_weights, rays_mask, projected_pos_maps):
        """
        Take in source image features, interpolate and sample them onto the target image, and mask out non-visible features.
        """
        resolution = torch.Tensor([projected_pos_maps.shape[2], projected_pos_maps.shape[1]]).int()

        # Mask out invalid values and points at infinity. Get the positions of each of the target points in source views.
        projected_pos_maps[torch.isinf(projected_pos_maps)] = 0.
        projected_pos_maps[(projected_pos_maps != projected_pos_maps)] = 0.
        projected_pos_maps_valid = projected_pos_maps[..., :2] * frustum_mask.float().unsqueeze(-1) * \
                                   rays_mask.float().unsqueeze(-1)
        projected_pos_maps_xy_valid = sdf_rendering.ndc_to_standard(projected_pos_maps_valid, resolution)

        # Sample source features corresponding to these points, mask out non-visible points.
        image_features_sampled = F.grid_sample(source_image_features, projected_pos_maps_xy_valid, align_corners=True)
        image_features_sampled = image_features_sampled * frustum_mask.float().unsqueeze(1) * \
                                 occlusion_weights.float().unsqueeze(1) * rays_mask.unsqueeze(1)
        return image_features_sampled

    def get_ray_d_per_feature(self, projected_pos_maps, source_view_ids, view_matrix_custom=None, dataset_num=None,
                              vid_frame=0):
        """
        Get the ray angles arriving at each of the projected points. These are used to weigh the features.
        """
        ray_builder_curr = self.get_ray_builder(dataset_num)

        rays_d_per_feature = torch.zeros_like(projected_pos_maps)

        projected_pos_maps[torch.isinf(projected_pos_maps)] = 0.

        for i, view_id in enumerate(source_view_ids):
            proj_matrix = ray_builder_curr.img_dataset.frames[0].image_views[view_id].projection_matrix
            view_matrix = ray_builder_curr.img_dataset.frames[0].image_views[view_id].view_matrix
            model_matrix = ray_builder_curr.model_matrix

            # Convert projected position maps to homogeneous coordinates.
            projected_pos_maps_homog = projected_pos_maps[i].view(-1, 3)
            projected_pos_maps_homog = torch.cat([projected_pos_maps_homog, torch.ones_like(projected_pos_maps_homog)[..., 0].unsqueeze(-1)], -1)

            # Inverse transform from NDC pixels to camera space.
            screen_coords_in_cam_space = mut.transform_vectors(torch.inverse(proj_matrix), projected_pos_maps_homog)

            # Build ray directions in camera space.
            rays_d_cam = mut.normalize_vecs(screen_coords_in_cam_space[..., :3])

            # Apply camera rotation to get world coordinates.
            model_view_matrix = view_matrix @ model_matrix
            rays_d = mut.transform_vectors(torch.inverse(model_view_matrix)[..., :3, :3], rays_d_cam)
            rays_d = mut.normalize_vecs(rays_d)

            rays_d_per_feature[i] = rays_d.view(projected_pos_maps.shape[1], projected_pos_maps.shape[2], -1)

        if view_matrix_custom is not None:
            proj_matrix = ray_builder_curr.img_dataset.frames[0].image_views[0].projection_matrix
            view_matrix = view_matrix_custom
            model_matrix = ray_builder_curr.model_matrix

            # Convert projected position maps to homogeneous coordinates.
            projected_pos_maps_homog = projected_pos_maps[-1].view(-1, 3)
            projected_pos_maps_homog = torch.cat([projected_pos_maps_homog, torch.ones_like(projected_pos_maps_homog)[..., 0].unsqueeze(-1)], -1)

            # Inverse transform from NDC pixels to camera space.
            screen_coords_in_cam_space = mut.transform_vectors(torch.inverse(proj_matrix), projected_pos_maps_homog)

            # Build ray directions in camera space.
            rays_d_cam = mut.normalize_vecs(screen_coords_in_cam_space[..., :3])

            # Apply camera rotation to get world coordinates.
            model_view_matrix = view_matrix @ model_matrix
            rays_d = mut.transform_vectors(torch.inverse(model_view_matrix)[..., :3, :3], rays_d_cam)
            rays_d = mut.normalize_vecs(rays_d)

            rays_d_per_feature[-1] = rays_d.view(projected_pos_maps.shape[1], projected_pos_maps.shape[2], -1)

        return rays_d_per_feature

    def aggregate_features_sum(self, image_features, projected_pos_maps=None, source_view_ids=None,
                               view_matrix_custom=None):
        """
        Aggregate features into a single feature, based on just summing visible features.
        """
        return image_features.sum(0, keepdim=True), None

    def aggregate_features_mean(self, image_features, projected_pos_maps=None, source_view_ids=None):
        """
        Aggregate features into a single feature, by taking the mean of the visible features.
        """
        # Get number of nonzero features. If none are visible, set to 1. to avoid getting 0/0 nans in image features.
        with torch.no_grad():
            nonzero_features = (image_features != 0).float()
            nonzero_features_sum = nonzero_features.sum(0, keepdim=True)
            nonzero_features_sum[nonzero_features_sum == 0] = 1.

        image_features_mean = image_features.sum(0, keepdim=True) / nonzero_features_sum
        return image_features_mean, None

    def aggregate_features_lumigraph(self, image_features, projected_pos_maps_a, source_view_ids):
        """
        Aggregate features into a single feature, weighting features by cosine similarity with the viewing direction
        of the target camera.
        """
        # Get ray directions from each of the projected positions.
        ray_d_a = self.get_ray_d_per_feature(projected_pos_maps_a, source_view_ids)

        with torch.no_grad():
            # Compute weights based on viewing direction:
            view_d = ray_d_a[-1]
            cos_similarity = (ray_d_a * view_d.unsqueeze(0)).sum(-1)
            cos_similarity = F.relu(cos_similarity)
            cos_similarity = cos_similarity[:-1]

            # Get number of nonzero features. If none are visible, set to 1. to avoid getting 0/0 nans in image features.
            nonzero_features = (image_features.sum(1) != 0).float()

        # Normalize features:
        cos_similarity *= nonzero_features
        cos_similarity_sum = cos_similarity.sum(0, keepdim=True)
        cos_similarity_sum[cos_similarity_sum == 0] = 1.
        cos_similarity = cos_similarity / cos_similarity_sum

        return (image_features * cos_similarity.unsqueeze(1)).sum(0, keepdim=True), cos_similarity

    def aggregate_features_lumigraph_epipolar(self, image_features, projected_pos_maps_a, source_view_ids,
                                              view_matrix_custom=None, dataset_num=None, params=None,
                                              occlusion_disp=None, vid_frame=0):
        """
        Aggregate features into a single feature, weighting features by the lumigraph method presented in
        Neural Lumigraph rendering.
        """
        # Get ray directions from each of the projected positions.
        ray_d_a = self.get_ray_d_per_feature(projected_pos_maps_a, source_view_ids, view_matrix_custom=view_matrix_custom,
                                             dataset_num=dataset_num, vid_frame=vid_frame)

        with torch.no_grad():
            # Compute viewing angle between ray towards target view and source view.
            view_d = ray_d_a[-1]
            tau = torch.acos(torch.clamp((ray_d_a * view_d.unsqueeze(0)).sum(-1), min=-1, max=1))
            tau = tau[:-1]

            # Get number of nonzero features. If none are visible, set to 1. to avoid getting 0/0 nans in image features.
            nonzero_features = (image_features.sum(1) != 0).float()

        # Get features using equation, and mask out invisible features
        w_hat = (1./tau) * (1 - (tau/tau.max(dim=0, keepdims=True)[0]))
        w_hat *= nonzero_features
        w_hat[w_hat != w_hat] = 0  # remove inf*0 NaN

        # Normalize features
        w_hat_sum = w_hat.sum(0, keepdim=True)
        w_hat_sum[w_hat_sum == 0] = 1.
        w = w_hat / w_hat_sum
        w[w != w] = 1.  # remove inf/inf NaN

        return (image_features * w.unsqueeze(1)).sum(0, keepdim=True), w

    def aggregate_features_learned(self, image_features, projected_pos_maps_a, source_view_ids,
                                   view_matrix_custom=None, dataset_num=None, params=None,
                                   occlusion_disp=None, vid_frame=0):
        with torch.no_grad():
            # Get ray directions from each of the projected positions.
            ray_d_a = self.get_ray_d_per_feature(projected_pos_maps_a, source_view_ids, view_matrix_custom=view_matrix_custom,
                                                 dataset_num=dataset_num, vid_frame=vid_frame)
            view_d = ray_d_a[-1:]

        image_features_reshape = image_features.permute(0, 2, 3, 1)
        view_d = view_d.expand(image_features_reshape.shape[0], -1, -1, -1)
        image_features_view_dirs = torch.cat([image_features_reshape, view_d], dim=-1)

        image_feature_weights = self.agg_net(image_features_view_dirs, params=get_subdict(params, 'agg_net'))

        # Get number of nonzero features. Mask out these weights, then normalize.
        nonzero_features = (image_features.sum(1) != 0).float().unsqueeze(-1)
        image_feature_weights = image_feature_weights * nonzero_features

        image_feature_weights_sum = image_feature_weights.sum(dim=0, keepdim=True)
        image_feature_weights_sum[image_feature_weights_sum == 0] = 1.
        w = image_feature_weights / image_feature_weights_sum

        return (image_features * w.permute(0, 3, 1, 2)).sum(0, keepdim=True), w.squeeze()

    def decode_image(self, target_feature, mask_in=None, params=None):
        """
        Decode feature grids into images.
        """
        return self.dec_net.forward(target_feature, params=get_subdict(params, 'dec_net'))

    def decode_image_rgb(self, x, mask_in=None, params=None):
        return (x / 2) + 0.5

    def source_view_select_random(self, target_view_id, dataset_num=None):
        if self.opt.is_meta:
            source_view_ids = np.random.choice(self.total_number_views[dataset_num]-1,
                                               self.opt.source_views_per_target[dataset_num], replace=False).tolist()
        else:
            source_view_ids = np.random.choice(self.total_number_views-1,
                                               self.opt.source_views_per_target, replace=False).tolist()
        source_view_ids = list(map(lambda x: x+1 if x >= target_view_id else x, source_view_ids)) + [target_view_id]

        return source_view_ids

    def source_view_select_random_withheld(self, target_view_id, dataset_num=None):
        if self.opt.is_meta:
            if self.opt.dataset_name == 'shapenet':
                TRAINVIEWS = self.opt.TRAIN_VIEWS.copy()
                if target_view_id in TRAINVIEWS:
                    TRAINVIEWS.remove(target_view_id)
                source_view_ids = TRAINVIEWS + [target_view_id]
                return source_view_ids
            else:
                source_view_ids = np.random.choice(self.opt.TRAIN_VIEWS.copy(), self.opt.source_views_per_target[dataset_num],
                                                   replace=False).tolist()
                if target_view_id in source_view_ids:
                    source_view_ids.remove(target_view_id)
                else:
                    del source_view_ids[-1]
                source_view_ids = source_view_ids + [target_view_id]
                return source_view_ids
        else:
            if self.opt.dataset_name == 'shapenet':
                TRAINVIEWS = self.opt.TRAIN_VIEWS.copy()
                if target_view_id in TRAINVIEWS:
                    TRAINVIEWS.remove(target_view_id)
                source_view_ids = TRAINVIEWS + [target_view_id]
                return source_view_ids
            else:
                source_view_ids = np.random.choice(self.opt.TRAIN_VIEWS.copy(), self.opt.source_views_per_target,
                                                   replace=False).tolist()
                if target_view_id in source_view_ids:
                    source_view_ids.remove(target_view_id)
                else:
                    del source_view_ids[-1]
                source_view_ids = source_view_ids + [target_view_id]
                return source_view_ids

    def source_view_select_nearest(self, target_view_id, dataset_num=None):
        if self.opt.is_meta:
            similarities = self.extrinsic_similarity_matrix_init[dataset_num][target_view_id]
            src_view_per_trgt = self.opt.source_views_per_target[dataset_num]
        else:
            similarities = self.extrinsic_similarity_matrix_init[target_view_id]
            src_view_per_trgt = self.opt.source_views_per_target
        similarities[target_view_id] = -1*float('inf')
        sorted, indices = torch.sort(similarities, descending=True)
        source_view_ids = indices[:src_view_per_trgt].cpu().numpy().tolist() + [target_view_id]

        return source_view_ids

    def forward(self, model_input, params=None, dataset_num=None):
        if self.opt.is_meta:
            model_input['train_shape'] = 1
        bs = self.opt.image_batch_size

        # Get target view and source view ids, and project target view to source views. Target is last ID.
        target_view_id = int(model_input.get('target_view_id', None))
        if target_view_id is None:
            target_view_id = int(model_input['rays_view_ids'][0,0].cpu().numpy())
        source_view_ids = self.source_view_select_fn(target_view_id, dataset_num)

        source_image_features_a, source_images_a = self.encoding_fn(source_view_ids, dataset_num, params=params)

        if model_input['train_shape']:
            computed_buffers_new = self.compute_3D_buffers(source_view_ids[-bs:], grad=1, params=params, dataset_num=dataset_num)
            computed_buffers_old = self.precomputed_buffers
            for key in computed_buffers_new.keys():
                computed_buffers_old[key] = computed_buffers_new[key]
            computed_buffers = computed_buffers_old
        else:
            computed_buffers = self.precomputed_buffers

        trgt_imgs = {}

        for target_view_idx, target_view_id in enumerate(source_view_ids[-bs:]):
            target_view_idx += len(source_view_ids[:-bs])
            single_output = self.forward_single(target_view_id, target_view_idx, source_view_ids, source_image_features_a,
                                                computed_buffers, params=params, dataset_num=dataset_num,
                                                return_dense=(target_view_idx == len(source_view_ids)-1),
                                                train_shape=model_input['train_shape'])

            trgt_imgs[target_view_idx] = single_output

        if model_input['train_shape']:
            # Prevent storing unnecessary stuff
            for key in computed_buffers_new.keys():
                computed_buffers_old[key] = {'pos': computed_buffers_new[key]['pos'].detach()}

        # Get SDF points to supervise priors on SDF
        sdf_out = self.decoder_sdf.forward(model_input, params=get_subdict(params, 'decoder_sdf'))['model_out']

        return {'trgt_outputs': trgt_imgs,
                'source_images': source_images_a,
                'sdf_out': sdf_out,
                'model_in': model_input,
                'dense_id': source_view_ids[-1],
                'dense_idx': len(source_view_ids)-1}

    def forward_single(self, target_view_id, target_view_idx, source_view_ids,
                       source_image_features_a, computed_buffers, params=None, dataset_num=None,
                       return_dense=False, train_shape=False):
        projected_pos_maps_a = self.project_target_view_to_sources(target_view_id, source_view_ids, computed_buffers,
                                                                   dataset_num=dataset_num)

        frustum_mask_a = self.get_frustum_mask(projected_pos_maps_a)
        occlusion_weights_a, occlusion_disp_a = self.occlusion_fn(projected_pos_maps_a, frustum_mask_a, target_view_id,
                                                   source_view_ids, computed_buffers, source_image_features_a,
                                                   dataset_num=dataset_num)
        rays_mask_a, proj_rays_mask_a = self.get_rays_mask(projected_pos_maps_a, frustum_mask_a, target_view_id,
                                                           source_view_ids, dataset_num=dataset_num)

        # Remove trgt idx
        frustum_mask = torch.cat([frustum_mask_a[:target_view_idx], frustum_mask_a[target_view_idx+1:]], 0)
        occlusion_weights = torch.cat([occlusion_weights_a[:target_view_idx], occlusion_weights_a[target_view_idx+1:]], 0)
        proj_rays_mask = torch.cat([proj_rays_mask_a[:target_view_idx], proj_rays_mask_a[target_view_idx+1:]], 0)
        projected_pos_maps = torch.cat([projected_pos_maps_a[:target_view_idx], projected_pos_maps_a[target_view_idx+1:]], 0)
        source_image_features = torch.cat([source_image_features_a[:target_view_idx], source_image_features_a[target_view_idx+1:]], 0)
        occlusion_disp = torch.cat([occlusion_disp_a[:target_view_idx], occlusion_disp_a[target_view_idx+1:]], 0)

        image_features_sampled = self.sample_features(source_image_features, frustum_mask, occlusion_weights,
                                                      proj_rays_mask, projected_pos_maps)

        # Modify source list for feature aggregation
        source_view_ids_mod = source_view_ids.copy()
        assert source_view_ids_mod.pop(target_view_idx) == target_view_id
        source_view_ids_mod += [target_view_id]

        aggregated_features, agg_weights = self.aggregation_fn(image_features_sampled, projected_pos_maps_a,
                                                               source_view_ids_mod, dataset_num=dataset_num, params=params,
                                                               occlusion_disp=occlusion_disp)

        dec_time = time.time()
        target_img = self.decoding_fn(aggregated_features, params=params)
        dec_time = time.time() - dec_time
        # print(f'Decoder Length: {dec_time}')

        if train_shape:
            trgt_rays_is_valid = computed_buffers[target_view_id]['mask']
            trgt_rays_sdf = computed_buffers[target_view_id]['sdf_raytraced']
            # min_rays_sdf = computed_buffers[target_view_id]['sdf_min']
            trgt_rays_softmask = computed_buffers[target_view_id]['softmask']
        else:
            trgt_rays_is_valid = None
            trgt_rays_sdf = None
            # min_rays_sdf = None
            trgt_rays_softmask = None

        if return_dense:
            return {'target_id': target_view_id,
                    'target_img': target_img,
                    'frustum_mask': frustum_mask_a,
                    'occlusion_weights': occlusion_weights_a,
                    'rays_mask': rays_mask_a,
                    'proj_rays_mask': proj_rays_mask_a,
                    'aggragation_weights': agg_weights,
                    'aggregated_features': aggregated_features,
                    'projected_pos_maps': projected_pos_maps_a,
                    'trgt_rays_is_valid': trgt_rays_is_valid,
                    'trgt_rays_sdf': trgt_rays_sdf,
                    # 'min_rays_sdf': min_rays_sdf,
                    'softmask': trgt_rays_softmask
                    }

        return {'target_img': target_img,
                'target_id': target_view_id,
                'rays_mask': rays_mask_a,
                'trgt_rays_is_valid': trgt_rays_is_valid,
                'trgt_rays_sdf': trgt_rays_sdf,
                # 'min_rays_sdf': min_rays_sdf,
                'softmask': trgt_rays_softmask
                }

    @torch.no_grad()
    def forward_test(self, target_view_id):
        """
        Generate output image.
        """
        # Get target view and source view ids, and project target view to source views. Target is last ID.
        source_view_ids = self.source_view_select_fn(target_view_id)
        print(source_view_ids)

        source_image_features_a, source_images_a = self.encoding_fn(source_view_ids)

        computed_buffers = self.precomputed_buffers
        projected_pos_maps_a = self.project_target_view_to_sources(target_view_id, source_view_ids, computed_buffers)

        # Check to see if these points are in the viewing range of the other views. This generates the three masks. Getting these masks is non differentiable.
        frustum_mask_a = self.get_frustum_mask(projected_pos_maps_a)
        occlusion_weights_a, _ = self.occlusion_fn(projected_pos_maps_a, frustum_mask_a, target_view_id,
                                                   source_view_ids, computed_buffers, source_image_features_a)
        rays_mask_a, proj_rays_mask_a = self.get_rays_mask(projected_pos_maps_a, frustum_mask_a, target_view_id, source_view_ids)

        # Remove masks / positions of the target view ID.
        frustum_mask = frustum_mask_a[:-1]
        occlusion_weights = occlusion_weights_a[:-1]
        proj_rays_mask = proj_rays_mask_a[:-1]
        projected_pos_maps = projected_pos_maps_a[:-1]
        source_image_features = source_image_features_a[:-1]

        # Sample image features for each other image and mask out non-visible features from target view.
        image_features_sampled = self.sample_features(source_image_features, frustum_mask, occlusion_weights, proj_rays_mask, projected_pos_maps)

        # Aggregate features.
        aggregated_features, agg_weights = self.aggregation_fn(image_features_sampled, projected_pos_maps_a, source_view_ids)

        # Decode image features into an image.
        target_img = self.decoding_fn(aggregated_features)

        return {'target_img': target_img,
                'mask': rays_mask_a[-1]}

    @torch.no_grad()
    def select_nearest_source_views(self, view_matrix):
        """
        Same as previous method, but computes nearest views given a view matrix.
        """
        extrinsics = []
        for view_id in range(self.total_number_views):
            extrinsic = self.ray_builder.img_dataset.frames[0].image_views[view_id].extrinsics
            extrinsics += [extrinsic]
        extrinsics = torch.stack(extrinsics, 0)

        r_quat = torch.from_numpy(R.from_matrix(view_matrix[:3, :3].cpu()).as_quat()).float()
        t_vec = torch.from_numpy(view_matrix[:3, 3].cpu().numpy())
        extrinsic_test = torch.cat((r_quat, t_vec)).to(self.opt.device)

        similarities = torch.matmul(extrinsics, extrinsic_test.unsqueeze(-1)).squeeze()
        sorted, indices = torch.sort(similarities, descending=True)
        source_view_ids = indices[:self.opt.source_views_per_target+1].cpu().numpy().tolist()

        return source_view_ids

    @torch.no_grad()
    def forward_test_custom(self, model_matrix, view_matrix, projection_matrix, resolution, vid_frame=0):
        # Get target view and source view ids, and project target view to source views.
        source_view_ids = self.source_view_select_fn(0)
        print(source_view_ids)

        source_image_features, source_images = self.encoding_fn(source_view_ids)

        computed_buffers = self.precomputed_buffers
        trgt_res = sdf_rendering.render_view_proj(self, torch.from_numpy(resolution).cuda(), model_matrix.cuda(),
                                                  view_matrix.cuda(), projection_matrix.cuda(), -1, batch_size=0,
                                                  vid_frame=vid_frame)
        projected_pos_maps_a = self.project_target_view_to_sources(trgt_res, source_view_ids)

        valid_mask = trgt_res['mask']

        trgt_res_proj_pos = self.project_view(trgt_res['pos'], model_matrix, view_matrix, projection_matrix)
        projected_pos_maps_a = torch.cat([projected_pos_maps_a, trgt_res_proj_pos.unsqueeze(0)], dim=0)

        # Check to see if these points are in the viewing range of the other views. This generates the three masks. Getting these masks is non differentiable.
        frustum_mask_a = self.get_frustum_mask(projected_pos_maps_a)
        occlusion_weights_a, _ = self.occlusion_fn(projected_pos_maps_a, frustum_mask_a, trgt_res,
                                                   source_view_ids, computed_buffers)
        rays_mask_a, proj_rays_mask_a = self.get_rays_mask(projected_pos_maps_a, frustum_mask_a, trgt_res, source_view_ids)

        frustum_mask = frustum_mask_a[:-1]
        occlusion_weights = occlusion_weights_a[:-1]
        proj_rays_mask = proj_rays_mask_a[:-1]
        projected_pos_maps = projected_pos_maps_a[:-1]

        # Sample image features for each other image and mask out non-visible features from target view.
        image_features_sampled = self.sample_features(source_image_features, frustum_mask, occlusion_weights, proj_rays_mask, projected_pos_maps)

        # Aggregate features.
        aggregated_features, agg_weights = self.aggregation_fn(image_features_sampled, projected_pos_maps_a, source_view_ids,
                                                               view_matrix_custom=view_matrix, vid_frame=vid_frame)

        # Decode image features into an image.
        target_img = self.decoding_fn(aggregated_features)

        return {'target_img': target_img,
                'source_imgs': source_images,
                'agg_weights': agg_weights,
                'valid_mask': valid_mask}

    def forward_with_activations(self, model_input):
        """
        Concatenate outputs of all subnets.
        Returns not only model output, but also intermediate activations.
        """
        raise RuntimeError("Not implemented.")

    def load_checkpoint(self, checkpoint_file, load_sdf=False, strict=False,
                        load_img_encoder=False, load_img_decoder=False, load_aggregation=False, load_poses=False):
        """
        Loads checkpoint.
        """
        print(
            f'Loading checkpoint from {checkpoint_file} (load_sdf={load_sdf}, load_img_encoder'
            f'={load_img_encoder}, load_img_decoder={load_img_decoder}, load_aggregation={load_aggregation}, load_poses={load_poses}).')
        state = torch.load(checkpoint_file, map_location=self.device)
        state_filtered = {}
        for k, v in state.items():
            if not load_sdf and k.startswith('decoder_sdf'):
                # Skip sdf.
                continue
            if not load_img_encoder and k.startswith('enc_net'):
                continue
            if not load_img_decoder and k.startswith('dec_net'):
                continue
            if not load_aggregation and k.startswith('agg_net'):
                continue
            if not load_poses and k.startswith('ray_builder'):
                continue
            state_filtered[k] = v
        self.load_state_dict(state_filtered, strict=strict)

    def gaussian_smooth(self, input, channels, kernel_size, sigma):
        """
        Gaussian smooth some input image
        """
        dim = 2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        return F.conv2d(input, weight=kernel.to(self.device), groups=channels)

    @property
    def device(self):
        """
        CUDA or CPU?
        """
        return self.decoder_sdf.device

    def renormalize(self):
        """
        Normalizes parameters.
        """
        if self.ray_builder is not None:
            if self.opt.is_meta:
                for rb in self.ray_builder:
                    rb['ray_builder'].renormalize()
            else:
                self.ray_builder.renormalize()
