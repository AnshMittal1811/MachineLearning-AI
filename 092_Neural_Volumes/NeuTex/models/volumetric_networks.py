import torch
import torch.nn as nn
import numpy as np

from .point_decoder import GaussianPointDecoder
from .nerf_decoders import RadianceDecoder, MlpDecoder
from .diff_render_func import (
    find_render_function,
    find_blend_function,
    simple_tone_map,
)
from .diff_ray_marching import (
    find_ray_generation_method,
    find_refined_ray_generation_method,
    ray_march,
    alpha_ray_march,
)
from .nv_decoders import VolumeDecoder, WarpDecoder
from .nv_mixed_decoder import MixedDecoder, MixedSeparatedDecoder
from .encoders import ConvEncoder, GaussianEmbedding
from utils import format as fmt
import time


def create_raymarching_model(opt):
    if opt.which_raymarching_model == "hierarchical":
        return HierarchicalRayMarching(opt)
    raise RuntimeError("Unknown ray marching model: " + opt.which_raymarching_model)


class HierarchicalRayMarching(nn.Module):
    def check_opt(self, opt):
        # sanity check
        if opt.which_decoder_model in ["mlp", "nv_mlp", "nv_mixed"]:
            if opt.which_render_func not in ["microfacet"]:
                print(
                    fmt.RED
                    + "[Big warning] mlp type decoder should use microfacet based rendering"
                    + fmt.END
                )
                time.sleep(3)
            if opt.which_blend_func not in ["alpha2"]:
                print(
                    fmt.YELLOW
                    + "[Warning] mlp type decoder typically uses alpha2 blending, using alpha blending effectively ignores the attenuation of light by the volume"
                    + fmt.END
                )
                time.sleep(3)
        if opt.which_decoder_model == "radiance":
            if opt.which_render_func not in ["radiance"]:
                print(
                    fmt.RED
                    + "[Big warning] Radiance based decoder should use radiance render function"
                    + fmt.END
                )
                time.sleep(3)
            if opt.which_blend_func not in ["alpha"]:
                print(
                    fmt.RED
                    + "[Big warning] Radiance based decoder should use alpha blending"
                    + fmt.END
                )
                time.sleep(3)

        if opt.embedding_model != "none":
            assert (
                opt.encoder_model == "none"
            ), "Cannot use embedding and encoder together"
            assert opt.embedding_size > 0
            if opt.embedding_model == "gaussian":
                assert (
                    opt.loss_kld_weight > 0
                ), "Gaussian encoder should be normalized with KLD"

        if opt.encoder_model != "none":
            assert opt.encoder_width > 0
            assert opt.encoder_channels > 0
            assert opt.loss_kld_weight > 0, "Encoder should be normalized with KLD"

    def __init__(self, opt):
        super(HierarchicalRayMarching, self).__init__()
        self.coarse_sample_num = opt.coarse_sample_num
        self.fine_sample_num = opt.fine_sample_num
        self.out_channels = opt.out_channels
        self.num_pos_freqs = opt.num_pos_freqs
        self.num_viewdir_freqs = opt.num_viewdir_freqs
        self.perturb = opt.perturb
        self.domain_size = opt.domain_size

        self.check_opt(opt)

        # ray generation
        self.raygen = find_ray_generation_method(opt.which_ray_generation)
        if opt.fine_sample_num > 0:
            self.fine_raygen = find_refined_ray_generation_method(
                opt.which_ray_generation
            )

        if opt.encoder_model == "vae":
            self.latent_model = "vae"
            self.encoder = ConvEncoder(
                opt.encoder_channels,
                opt.encoder_width,
                opt.embedding_size,
                opt.encoder_normalization > 0,
            )
            self.down = nn.Sequential(
                nn.Linear(opt.embedding_size, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
            )
        elif opt.embedding_model == "l2":
            self.latent_model = "l2_embedding"
            self.embedding_model = nn.Embedding(opt.num_embeddings, opt.embedding_size)
            self.down = nn.Sequential(
                nn.Linear(opt.embedding_size, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
            )
        elif opt.embedding_model == "gaussian":
            self.latent_model = "gaussian_embedding"
            self.embedding_model = GaussianEmbedding(
                opt.num_embeddings, opt.embedding_size
            )
            self.down = nn.Sequential(
                nn.Linear(opt.embedding_size, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
            )
        else:
            self.latent_model = "none"
            self.embedding = nn.Parameter(torch.randn((1, 256)).float())

        if opt.loss_depth_weight > 0 or opt.loss_mask_weight > 0 or not opt.is_train:
            self.return_depth = True
        else:
            self.return_depth = False

        if opt.loss_color_weight > 0:
            self.return_color = True
        else:
            self.return_color = False

        # warp
        self.warp_model = WarpDecoder(
            256, opt.which_warp_model, opt.use_displacement_warp > 0
        )
        if opt.add_global_warp:
            self.global_warp_model = WarpDecoder(
                256, "global", opt.use_displacement_warp > 0
            )
        else:
            self.global_warp_model = None

        # decoder
        if opt.which_decoder_model == "mlp":
            self.net_coarse_decoder = MlpDecoder(
                num_freqs=self.num_pos_freqs,
                out_channels=self.out_channels,
                encoding_size=256 if self.latent_model != "none" else 0,
            )
            if opt.fine_sample_num > 0:
                self.net_fine_decoder = MlpDecoder(
                    num_freqs=self.num_pos_freqs,
                    out_channels=self.out_channels,
                    encoding_size=256 if self.latent_model != "none" else 0,
                )
        elif opt.which_decoder_model == "radiance":
            self.net_coarse_decoder = RadianceDecoder(
                num_freqs=self.num_pos_freqs,
                num_viewdir_freqs=self.num_viewdir_freqs,
                encoding_size=256 if self.latent_model != "none" else 0,
            )
            if opt.fine_sample_num > 0:
                self.net_fine_decoder = RadianceDecoder(
                    num_freqs=self.num_pos_freqs,
                    num_viewdir_freqs=self.num_viewdir_freqs,
                    encoding_size=256 if self.latent_model != "none" else 0,
                )
        elif opt.which_decoder_model == "nv_mlp":
            self.net_coarse_decoder = VolumeDecoder(
                256,
                template_type=opt.nv_template_type,
                template_res=opt.nv_resolution,
                out_channels=opt.out_channels,
            )
            if opt.fine_sample_num > 0:
                self.net_fine_decoder = VolumeDecoder(
                    256,
                    template_type=opt.nv_template_type,
                    template_res=opt.nv_resolution,
                    out_channels=opt.out_channels,
                )
        elif opt.which_decoder_model == "mixed_mlp":
            self.net_coarse_decoder = MixedDecoder(
                256,
                template_type=opt.nv_template_type,
                template_res=opt.nv_resolution,
                mlp_channels=128,
                out_channels=opt.out_channels,
                position_freqs=opt.num_pos_freqs,
                feature_freqs=opt.num_feature_freqs,
            )
            if opt.fine_sample_num > 0:
                self.net_fine_decoder = MixedDecoder(
                    256,
                    template_type=opt.nv_template_type,
                    template_res=opt.nv_resolution,
                    mlp_channels=128,
                    out_channels=opt.out_channels,
                    position_freqs=opt.num_pos_freqs,
                    feature_freqs=opt.num_feature_freqs,
                )
        elif opt.which_decoder_model == "mixed_separate_code":
            self.net_coarse_decoder = MixedSeparatedDecoder(
                256,
                template_type=opt.nv_template_type,
                template_res=opt.nv_resolution,
                mlp_channels=128,
                out_channels=opt.out_channels,
                position_freqs=opt.num_pos_freqs,
                feature_freqs=opt.num_feature_freqs,
            )
            if opt.fine_sample_num > 0:
                self.net_fine_decoder = MixedSeparatedDecoder(
                    256,
                    template_type=opt.nv_template_type,
                    template_res=opt.nv_resolution,
                    mlp_channels=128,
                    out_channels=opt.out_channels,
                    position_freqs=opt.num_pos_freqs,
                    feature_freqs=opt.num_feature_freqs,
                )
        elif opt.which_decoder_model == "gaussian_point":
            self.net_coarse_decoder = GaussianPointDecoder(
                128, 128, out_channels=opt.out_channels
            )
            if opt.fine_sample_num > 0:
                self.net_fine_decoder = GaussianPointDecoder(
                    128, 128, out_channels=opt.out_channels
                )

        else:
            raise RuntimeError("Unknown decoder model: " + opt.which_decoder_model)

        # ray marching
        self.render_func = find_render_function(opt.which_render_func)
        self.blend_func = find_blend_function(opt.which_blend_func)

    def build_point_cloud_visualization(
        self, data_idx, resolution=256, encoder_image=None, patch_size=65536 * 16
    ):
        with torch.no_grad():
            result = np.zeros((resolution * resolution * resolution, self.out_channels))

            assert (
                not self.training
            ), "Point cloud visualization should only be built at testing time"
            device = next(self.parameters()).device
            grid = np.stack(
                np.meshgrid(*([np.arange(resolution)] * 3), indexing="ij"), axis=-1
            )
            grid = (2 * grid + 1) / resolution - 1
            grid *= self.domain_size  # (res, res, res, 3)
            grid = torch.FloatTensor(grid).to(device)
            all_raypos = grid.view(1, -1, 1, 3)

            from tqdm import trange

            for start in trange(0, all_raypos.shape[1], patch_size):
                assert all_raypos.shape[1] % patch_size == 0
                raypos = all_raypos[:, start : start + patch_size]

                if self.latent_model == "vae":
                    mu, logstd2 = self.encoder(encoder_image)
                    embedding = self.encoder.sample(mu, logstd2)
                    embedding2 = self.down(embedding)
                elif self.latent_model == "l2_embedding":
                    embedding = self.embedding_model(data_idx)
                    embedding2 = self.down(embedding)
                elif self.latent_model == "gaussian_embedding":
                    mu, logstd2 = self.embedding_model(data_idx)
                    embedding = self.embedding_model.sample(mu, logstd2)
                    embedding2 = self.down(embedding)
                else:
                    embedding2 = self.embedding.expand(
                        (raypos.shape[0], self.embedding.shape[1])
                    )

                # warp ray pos
                if self.global_warp_model is not None:
                    raypos = self.global_warp_model(embedding2, raypos)
                raypos = self.warp_model(embedding2, raypos)
                # ray_valid = torch.prod((raypos > -1.) * (raypos < 1.),
                #                        dim=-1).byte()

                # decoding
                decoded_features = self.net_coarse_decoder(
                    raypos, None, encoding=embedding2
                )
                features = decoded_features.detach().cpu().numpy()
                result[start : start + patch_size] = features.reshape(
                    (-1, self.out_channels)
                )
            return (
                grid.data.cpu().numpy(),
                result.reshape((resolution, resolution, resolution, self.out_channels)),
            )

    def forward(
        self,
        campos,
        raydir,
        data_idx=None,
        lightpos=None,
        light_intensity=None,
        gt_image=None,
        encoder_image=None,
    ):
        # campos: N x 3
        # raydir: N x Ray x 3
        output = {}

        # ray generation
        raypos, ray_dist, ray_valid, ray_ts = self.raygen(
            campos,
            raydir,
            self.coarse_sample_num,
            domain_size=self.domain_size,
            jitter=0.05 if self.perturb > 0 else 0,
        )
        # raypos: (N, Rays, Samples, 3)

        # generate embedding
        if self.latent_model in ["l2_embedding", "gaussian_embedding"]:
            assert data_idx is not None, "Missing data_idx input required for embedding"
        if self.latent_model == "vae":
            assert (
                encoder_image is not None
            ), "Missing encoder_image required by encoder"

        kld = None
        embedding = None
        if self.latent_model == "vae":
            mu, logstd2 = self.encoder(encoder_image)
            embedding = self.encoder.sample(mu, logstd2)
            embedding2 = self.down(embedding)
            kld = self.encoder.kld(mu, logstd2)
        elif self.latent_model == "l2_embedding":
            embedding = self.embedding_model(data_idx)
            embedding2 = self.down(embedding)
        elif self.latent_model == "gaussian_embedding":
            mu, logstd2 = self.embedding_model(data_idx)
            embedding = self.embedding_model.sample(mu, logstd2)
            embedding2 = self.down(embedding)
            kld = self.embedding_model.kld(mu, logstd2)
        else:
            embedding2 = self.embedding.expand(
                (raypos.shape[0], self.embedding.shape[1])
            )

        # warp ray pos
        if self.global_warp_model is not None:
            raypos = self.global_warp_model(embedding2, raypos)
        raypos = self.warp_model(embedding2, raypos)
        valid = torch.prod((raypos > -1.0) * (raypos < 1.0), dim=-1).byte()
        ray_valid = ray_valid * valid

        # decoding
        decoded_features = self.net_coarse_decoder(
            raypos, (raydir[:, :, None, :]).expand_as(raypos), encoding=embedding2
        )

        # ray march
        if self.return_color:
            (
                ray_color,
                point_color,
                opacity,
                acc_transmission,
                blend_weight,
                background_transmission,
                _,
            ) = ray_march(
                raydir,
                raypos,
                ray_dist,
                ray_valid,
                decoded_features,
                lightpos,
                light_intensity,
                self.render_func,
                self.blend_func,
            )
            ray_color = simple_tone_map(ray_color)
            output["coarse_raycolor"] = ray_color
            output["coarse_point_opacity"] = opacity
        else:
            (
                opacity,
                acc_transmission,
                blend_weight,
                background_transmission,
                _,
            ) = alpha_ray_march(
                raydir, raypos, ray_dist, ray_valid, decoded_features, self.blend_func
            )

        if self.return_depth:
            alpha_blend_weight = opacity * acc_transmission
            weight = alpha_blend_weight.view(alpha_blend_weight.shape[:3])
            avg_depth = (weight * ray_ts).sum(-1) / (weight.sum(-1) + 1e-6)
            # avg_depth = (weight * ray_ts).sum(-1)   # NOTE: no weighted average
            output["coarse_depth"] = avg_depth
            output["coarse_is_background"] = background_transmission

        if self.fine_sample_num > 0:
            raypos, ray_dist, ray_valid, ray_ts = self.fine_raygen(
                campos,
                raydir,
                self.fine_sample_num,
                ray_ts,
                blend_weight,
                domain_size=self.domain_size,
                jitter=self.perturb > 0,
            )

            if self.global_warp_model is not None:
                raypos = self.global_warp_model(embedding2, raypos)
            raypos = self.warp_model(embedding2, raypos)
            valid = torch.prod((raypos > -1.0) * (raypos < 1.0), dim=-1).byte()
            ray_valid = ray_valid * valid

            decoded_features = self.net_fine_decoder(
                raypos, (raydir[:, :, None, :]).expand_as(raypos), encoding=embedding2
            )

            if self.return_color:
                (
                    ray_color,
                    point_color,
                    opacity,
                    acc_transmission,
                    blend_weight,
                    _,
                    _,
                ) = ray_march(
                    raydir,
                    raypos,
                    ray_dist,
                    ray_valid,
                    decoded_features,
                    lightpos,
                    light_intensity,
                    self.render_func,
                    self.blend_func,
                )
                ray_color = simple_tone_map(ray_color)
                output["fine_raycolor"] = ray_color
                output["fine_point_opacity"] = opacity
            else:
                (
                    opacity,
                    acc_transmission,
                    blend_weight,
                    background_transmission,
                    _,
                ) = alpha_ray_march(
                    raydir,
                    raypos,
                    ray_dist,
                    ray_valid,
                    decoded_features,
                    self.blend_func,
                )

            if self.return_depth:
                alpha_blend_weight = opacity * acc_transmission
                weight = alpha_blend_weight.view(alpha_blend_weight.shape[:3])
                avg_depth = (weight * ray_ts).sum(-1) / (weight.sum(-1) + 1e-6)
                # avg_depth = (weight * ray_ts).sum(-1)  # NOTE: no weighted average
                output["fine_depth"] = avg_depth
                output["fine_is_background"] = background_transmission

        if kld is not None:
            output["kld"] = kld
        if embedding is not None:
            output["embedding"] = embedding
        return output

    def forward_with_interpolation(
        self,
        campos,
        raydir,
        data_idx=None,
        lightpos=None,
        light_intensity=None,
        mix_weight=None,
    ):
        # campos: N x 3
        # raydir: N x Ray x 3
        output = {}

        # ray generation
        raypos, ray_dist, ray_valid, ray_ts = self.raygen(
            campos,
            raydir,
            self.coarse_sample_num,
            domain_size=self.domain_size,
            jitter=0.05 if self.perturb > 0 else 0,
        )
        # raypos: (N, Rays, Samples, 3)

        assert data_idx is not None
        assert self.latent_model in ["l2_embedding", "gaussian_embedding"]

        assert data_idx.shape == (1, 2)
        assert mix_weight.shape == (1, 2)

        if self.latent_model == "l2_embedding":
            embedding = (self.embedding_model(data_idx) * mix_weight[:, :, None]).sum(
                -2
            )
            embedding2 = self.down(embedding)
        elif self.latent_model == "gaussian_embedding":
            mu, logstd2 = self.embedding_model(data_idx)
            embedding = (
                self.embedding_model.sample(mu, logstd2) * mix_weight[:, :, None]
            ).sum(-2)
            embedding2 = self.down(embedding)

        # warp ray pos
        if self.global_warp_model is not None:
            raypos = self.global_warp_model(embedding2, raypos)
        raypos = self.warp_model(embedding2, raypos)
        valid = torch.prod((raypos > -1.0) * (raypos < 1.0), dim=-1).byte()
        ray_valid = ray_valid * valid

        # decoding
        decoded_features = self.net_coarse_decoder(
            raypos, (raydir[:, :, None, :]).expand_as(raypos), encoding=embedding2
        )

        # ray march
        if self.return_color:
            (
                ray_color,
                point_color,
                opacity,
                acc_transmission,
                blend_weight,
                background_transmission,
                _,
            ) = ray_march(
                raydir,
                raypos,
                ray_dist,
                ray_valid,
                decoded_features,
                lightpos,
                light_intensity,
                self.render_func,
                self.blend_func,
            )
            ray_color = simple_tone_map(ray_color)
            output["coarse_raycolor"] = ray_color
            output["coarse_point_opacity"] = opacity
        else:
            (
                opacity,
                acc_transmission,
                blend_weight,
                background_transmission,
                _,
            ) = alpha_ray_march(
                raydir, raypos, ray_dist, ray_valid, decoded_features, self.blend_func
            )

        if self.return_depth:
            alpha_blend_weight = opacity * acc_transmission
            weight = alpha_blend_weight.view(alpha_blend_weight.shape[:3])
            avg_depth = (weight * ray_ts).sum(-1) / (weight.sum(-1) + 1e-6)
            output["coarse_depth"] = avg_depth
            output["coarse_is_background"] = background_transmission

        if self.fine_sample_num > 0:
            raypos, ray_dist, ray_valid, ray_ts = self.fine_raygen(
                campos,
                raydir,
                self.fine_sample_num,
                ray_ts,
                blend_weight,
                domain_size=self.domain_size,
                jitter=self.perturb > 0,
            )

            if self.global_warp_model is not None:
                raypos = self.global_warp_model(embedding2, raypos)
            raypos = self.warp_model(embedding2, raypos)
            valid = torch.prod((raypos > -1.0) * (raypos < 1.0), dim=-1).byte()
            ray_valid = ray_valid * valid

            decoded_features = self.net_fine_decoder(
                raypos, (raydir[:, :, None, :]).expand_as(raypos), encoding=embedding2
            )

            if self.return_color:
                (
                    ray_color,
                    point_color,
                    opacity,
                    acc_transmission,
                    blend_weight,
                    _,
                    _,
                ) = ray_march(
                    raydir,
                    raypos,
                    ray_dist,
                    ray_valid,
                    decoded_features,
                    lightpos,
                    light_intensity,
                    self.render_func,
                    self.blend_func,
                )
                ray_color = simple_tone_map(ray_color)
                output["fine_raycolor"] = ray_color
                output["fine_point_opacity"] = opacity
            else:
                (
                    opacity,
                    acc_transmission,
                    blend_weight,
                    background_transmission,
                    _,
                ) = alpha_ray_march(
                    raydir,
                    raypos,
                    ray_dist,
                    ray_valid,
                    decoded_features,
                    self.blend_func,
                )

            if self.return_depth:
                alpha_blend_weight = opacity * acc_transmission
                weight = alpha_blend_weight.view(alpha_blend_weight.shape[:3])
                avg_depth = (weight * ray_ts).sum(-1) / (weight.sum(-1) + 1e-6)
                output["fine_depth"] = avg_depth
                output["fine_is_background"] = background_transmission

        if embedding is not None:
            output["embedding"] = embedding
        return output
