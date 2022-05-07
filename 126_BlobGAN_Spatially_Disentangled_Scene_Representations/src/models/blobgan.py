from __future__ import annotations

__all__ = ["BlobGAN"]

import random
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Tuple, Dict

import einops
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from cleanfid import fid
from einops import rearrange, repeat
from matplotlib import cm
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torchvision.utils import make_grid

from models import networks
from models.base import BaseModule
from utils import FromConfig, run_at_step, get_D_stats, G_path_loss, D_R1_loss, freeze, is_rank_zero, accumulate, \
    mixing_noise, pyramid_resize, splat_features_from_scores, rotation_matrix
import utils

# SPLAT_KEYS = ['spatial_style', 'xs', 'ys', 'covs', 'sizes']
SPLAT_KEYS = ['spatial_style', 'scores_pyramid']
_ = Image
_ = make_grid


@dataclass
class Lossλs:
    D_real: float = 1
    D_fake: float = 1
    D_R1: float = 5
    G: float = 1
    G_path: float = 2

    G_feature_mean: float = 10
    G_feature_variance: float = 10

    def __getitem__(self, key):
        return super().__getattribute__(key)


@dataclass(eq=False)
class BlobGAN(BaseModule):
    # Modules
    generator: FromConfig[nn.Module]
    layout_net: FromConfig[nn.Module]
    discriminator: FromConfig[nn.Module]
    # Module parameters
    dim: int = 256
    noise_dim: int = 512
    resolution: int = 128
    p_mixing_noise: float = 0.0
    n_ema_sample: int = 8
    freeze_G: bool = False
    # Optimization
    lr: float = 1e-3
    eps: float = 1e-5
    # Regularization
    D_reg_every: int = 16
    G_reg_every: int = 4
    path_len: float = 0
    # Loss parameters
    λ: FromConfig[Lossλs] = None
    # Logging
    log_images_every_n_steps: Optional[int] = 500
    log_timing_every_n_steps: Optional[int] = -1
    log_fid_every_n_steps: Optional[int] = -1
    log_grads_every_n_steps: Optional[int] = -1
    log_fid_every_epoch: bool = True
    fid_n_imgs: Optional[int] = 5000
    fid_stats_name: Optional[str] = None
    flush_cache_every_n_steps: Optional[int] = -1
    fid_num_workers: Optional[int] = 24
    valtest_log_all: bool = False
    accumulate: bool = True
    validate_gradients: bool = False
    ipdb_on_nan: bool = False
    # Input feature generation
    n_features_min: int = 3
    n_features_max: int = 5
    feature_splat_temp: int = 2
    spatial_style: bool = False
    ab_norm: float = 0.01
    feature_jitter_xy: float = 0.0
    feature_jitter_shift: float = 0.0
    feature_jitter_angle: float = 0.0

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.discriminator = networks.get_network(**self.discriminator)
        self.generator_ema = networks.get_network(**self.generator)
        self.generator = networks.get_network(**self.generator)
        self.layout_net_ema = networks.get_network(**self.layout_net)
        self.layout_net = networks.get_network(**self.layout_net)
        if self.freeze_G:
            self.generator.eval()
            freeze(self.generator)
        if self.accumulate:
            self.generator_ema.eval()
            freeze(self.generator_ema)
            accumulate(self.generator_ema, self.generator, 0)
            self.layout_net_ema.eval()
            freeze(self.layout_net_ema)
            accumulate(self.layout_net_ema, self.layout_net, 0)
        else:
            del self.generator_ema
            del self.layout_net_ema
        self.λ = Lossλs(**self.λ)
        self.sample_z = torch.randn(self.n_ema_sample, self.noise_dim)

    # Initialization and state management
    def on_train_start(self):
        super().on_train_start()
        # Validate parameters w.r.t. trainer (must be done here since trainer is not attached as property yet in init)
        assert self.log_images_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_images_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder. ' \
            f'Got {self.log_images_every_n_steps} and {self.trainer.log_every_n_steps}.'
        assert self.log_timing_every_n_steps < 0 or self.log_timing_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_images_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder'
        assert self.log_fid_every_n_steps < 0 or self.log_fid_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_fid_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder'

    def configure_optimizers(self) -> Union[optim, List[optim]]:
        G_reg_ratio = self.G_reg_every / ((self.G_reg_every + 1) or -1)
        D_reg_ratio = self.D_reg_every / ((self.D_reg_every + 1) or -1)
        req_grad = lambda l: [p for p in l if p.requires_grad]
        decay_params = []
        G_params = [{'params': req_grad(self.generator.parameters()), 'weight_decay': 0}, {
            'params': [],
            'weight_decay': 0  # Legacy, dont remove :(
        }, {
                        'params': req_grad(
                            [p for p in self.layout_net.parameters() if not any([p is pp for pp in decay_params])]),
                        'weight_decay': 0
                    }]
        D_params = req_grad(self.discriminator.parameters())
        G_optim = torch.optim.AdamW(G_params, lr=self.lr * G_reg_ratio,
                                    betas=(0 ** G_reg_ratio, 0.99 ** G_reg_ratio), eps=self.eps, weight_decay=0)
        D_optim = torch.optim.AdamW(D_params, lr=self.lr * D_reg_ratio,
                                    betas=(0 ** D_reg_ratio, 0.99 ** D_reg_ratio), eps=self.eps, weight_decay=0)
        if is_rank_zero():
            print(f'Optimizing {sum([p.numel() for grp in G_params for p in grp["params"]]) / 1e6:.2f}M params for G '
                  f'and {sum([p.numel() for p in D_params]) / 1e6:.2f}M params for D')
        if self.freeze_G:
            return D_optim
        else:
            return G_optim, D_optim

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ):
        self.batch_idx = batch_idx
        optimizer.step(closure=optimizer_closure)

    def training_epoch_end(self, *args, **kwargs):
        try:
            if self.log_fid_every_epoch:
                self.log_fid("train")
        except:
            pass

    def gen(self, z=None, layout=None, ema=False, norm_img=False, ret_layout=False, ret_latents=False, noise=None,
            **kwargs):
        assert (z is not None) or (layout is not None)
        kwargs['return_metadata'] = ret_layout
        layout = self.generate_layout(z, metadata=layout, ema=ema, **kwargs)
        gen_input = {
            'input': layout['feature_grid'],
            'styles': {k: layout[k] for k in SPLAT_KEYS} if self.spatial_style else z,
            'return_image_only': not ret_latents,
            'return_latents': ret_latents,
            'noise': noise
        }
        G = self.generator_ema if ema else self.generator
        out = G(**gen_input)
        if norm_img:
            img = out[0] if ret_latents else out
            img.add_(1).div_(2).mul_(255)
        if ret_layout:
            if not ret_latents:
                out = [out]
            return [layout, *out]
        else:
            return out

    @torch.no_grad()
    def log_fid(self, mode, **kwargs):
        def gen_fn(z):
            return self.gen(z, ema=self.accumulate, norm_img=True)
        if is_rank_zero():
            dataset = self.trainer.datamodule.path.name
            fid_split = "custom"
            fid_score = fid.compute_fid(gen=gen_fn, dataset_name=self.fid_stats_name or f"lsun_{dataset}",
                                        dataset_res=self.resolution, num_gen=self.fid_n_imgs,
                                        dataset_split=fid_split, device=self.device,
                                        num_workers=self.fid_num_workers, z_dim=self.noise_dim)
        else:
            fid_score = 0.0
        fid_score = self.all_gather(fid_score).max().item()
        self.log_scalars({'fid': fid_score}, mode, **kwargs)

    # Training and evaluation
    @torch.no_grad()
    def visualize_features(self, xs, ys, viz_size, features=None, scores=None, feature_img=None,
                           c_border=-1, c_fill=1, sz=5, viz_entropy=False, viz_centers=False, viz_colors=None,
                           feature_center_mask=None, **kwargs) -> Dict[str, Tensor]:
        if feature_img is None:
            rand_colors = viz_colors is None
            viz_colors = (viz_colors if not rand_colors else torch.rand_like(features[..., :3])).to(xs.device)
            if viz_colors.ndim == 2:
                # viz colors should be [Kmax, 3]
                viz_colors = viz_colors[:features.size(1)][None].repeat_interleave(len(features), 0)
            elif viz_colors.ndim == 3:
                # viz colors should be [Nbatch, Kmax, 3]
                viz_colors = viz_colors[:, :features.size(1)]
            else:
                viz_colors = torch.rand_like(features[..., :3])
            img = splat_features_from_scores(scores, viz_colors, viz_size)
            if rand_colors:
                imax = img.amax((2, 3))[:, :, None, None]
                imin = img.amin((2, 3))[:, :, None, None]
                feature_img = img.sub(imin).div((imax - imin).clamp(min=1e-5)).mul(2).sub(1)
            else:
                feature_img = img
        imgs_flat = rearrange(feature_img, 'n c h w -> n c (h w)')
        if viz_centers:
            centers = torch.stack((xs, ys), -1).mul(viz_size).round()
            centers[..., 1].mul_(viz_size)
            centers = centers.sum(-1).long()
            if feature_center_mask is not None:
                fill_center = centers[torch.arange(len(centers)), feature_center_mask.int().argmax(1)]
                centers[~feature_center_mask] = fill_center.repeat_interleave((~feature_center_mask).sum(1), dim=0)
            offsets = (-sz // 2, sz // 2 + 1)
            offsets = (torch.arange(*offsets)[None] + torch.arange(*offsets).mul(viz_size)[:, None])
            border_mask = torch.zeros_like(offsets).to(bool)
            border_mask[[0, -1]] = border_mask[:, [0, -1]] = True
            offsets_border = offsets[border_mask].flatten()
            offsets_center = offsets[~border_mask].flatten()
            nonzero_features = scores[..., :-1].amax((1, 2)) > 0
            # draw center
            pixels = (centers[..., None] + offsets_center[None, None].to(self.device)) \
                .clamp(min=0, max=imgs_flat.size(-1) - 1)
            pixels = pixels.flatten(start_dim=1)
            pixels = repeat(pixels, 'n m -> n c m', c=3)
            empty_img = torch.ones_like(imgs_flat)
            imgs_flat.scatter_(dim=-1, index=pixels, value=c_fill)
            empty_img.scatter_(dim=-1, index=pixels, value=c_fill)
            # draw borders
            pixels = (centers[..., None] + offsets_border[None, None].to(self.device)) \
                .clamp(min=0, max=imgs_flat.size(-1) - 1)
            pixels = pixels.flatten(start_dim=1)
            pixels = repeat(pixels, 'n m -> n c m', c=3)
            imgs_flat.scatter_(dim=-1, index=pixels, value=c_border)
            empty_img.scatter_(dim=-1, index=pixels, value=c_border)
        out = {
            'feature_img': imgs_flat.reshape_as(feature_img)
        }
        if viz_centers:
            out['just_centers'] = empty_img.reshape_as(feature_img)
        if scores is not None and viz_entropy:
            img = (-scores.log2() * scores).sum(-1).nan_to_num(0)
            imax = img.amax((1, 2))[:, None, None]
            imin = img.amin((1, 2))[:, None, None]
            img = img.sub(imin).div((imax - imin).clamp(min=1e-5)).mul(256).int().cpu()
            h = w = img.size(-1)
            img = torch.from_numpy(cm.plasma(img.flatten())).mul(2).sub(1)[:, :-1]
            out['entropy_img'] = rearrange(img, '(n h w) c -> n c h w', h=h, w=w)
        return out

    def splat_features(self, xs: Tensor, ys: Tensor, features: Tensor, covs: Tensor, sizes: Tensor, size: int,
                       score_size: int, viz_size: Optional[int] = None, viz: bool = False,
                       return_metadata: bool = True,
                       covs_raw: bool = True, pyramid: bool = True, no_jitter: bool = False,
                       no_splat: bool = False, viz_score_fn=None,
                       **kwargs) -> Dict:
        """
        Args:
            xs: [N, M] X-coord location in [0,1]
            ys: [N, M] Y-coord location in [0,1]
            features: [N, M+1, dim] feature vectors to splat (and bg feature vector)
            covs: [N, M, 2, 2] xy covariance matrices for each feature
            sizes: [N, M+1] distributions of per feature (and bg) weights
            size: output grid size
            score_size: size at which to render score grid before downsampling to size
            viz_size: visualized grid in RGB dimension
            viz: whether to visualize
            covs_raw: whether covs already processed or not
            return_metadata: whether to return dict with metadata
            viz_score_fn: map from raw score to new raw score for generating blob maps. if you want to artificially enlarge blob borders, e.g., you can send in lambda s: s*1.5
            no_splat: return without computing scores, can be useful for visualizing
            no_jitter: manually disable jittering. useful for consistent results at test if model trained with jitter
            pyramid: generate score pyramid
            **kwargs: unused

        Returns: dict with requested information
        """
        if self.feature_jitter_xy and not no_jitter:
            xs = xs + torch.empty_like(xs).uniform_(-self.feature_jitter_xy, self.feature_jitter_xy)
            ys = ys + torch.empty_like(ys).uniform_(-self.feature_jitter_xy, self.feature_jitter_xy)
        if covs_raw:
            a, b = covs[..., :2].sigmoid().unbind(-1)
            ab_norm = 1
            if self.ab_norm is not None:
                ab_norm = self.ab_norm * (a * b).rsqrt()
            basis_i = covs[..., 2:]
            basis_i = F.normalize(basis_i, p=2, dim=-1)
            if self.feature_jitter_angle and not no_jitter:
                with torch.no_grad():
                    theta = basis_i[..., 0].arccos()
                    theta = theta + torch.empty_like(theta).uniform_(-self.feature_jitter_angle,
                                                                     self.feature_jitter_angle)
                    basis_i_jitter = (rotation_matrix(theta)[..., 0] - basis_i).detach()
                basis_i = basis_i + basis_i_jitter
            basis_j = torch.stack((-basis_i[..., 1], basis_i[..., 0]), -1)
            R = torch.stack((basis_i, basis_j), -1)
            covs = torch.zeros_like(R)
            covs[..., 0, 0] = a * ab_norm
            covs[..., -1, -1] = b * ab_norm
            covs = torch.einsum('...ij,...jk,...lk->...il', R, covs, R)
            covs = covs + torch.eye(2)[None, None].to(covs.device) * 1e-5

        if no_splat:
            return {'xs': xs, 'ys': ys, 'covs': covs, 'sizes': sizes, 'features': features}

        feature_coords = torch.stack((xs, ys), -1).mul(score_size)  # [n, m, 2]
        grid_coords = torch.stack(
            (torch.arange(score_size).repeat(score_size), torch.arange(score_size).repeat_interleave(score_size))).to(
            xs.device)  # [2, size*size]
        delta = (grid_coords[None, None] - feature_coords[..., None]).div(score_size)  # [n, m, 2, size*size]

        sq_mahalanobis = (delta * torch.linalg.solve(covs, delta)).sum(2)
        sq_mahalanobis = einops.rearrange(sq_mahalanobis, 'n m (s1 s2) -> n s1 s2 m', s1=score_size)

        # [n, h, w, m]
        shift = sizes[:, None, None, 1:]
        if self.feature_jitter_shift and not no_jitter:
            shift = shift + torch.empty_like(shift).uniform_(-self.feature_jitter_shift, self.feature_jitter_shift)
        scores = sq_mahalanobis.div(-1).add(shift).sigmoid()

        bg_scores = torch.ones_like(scores[..., :1])
        scores = torch.cat((bg_scores, scores), -1)  # [n, h, w, m+1]

        # alpha composite
        rev = list(range(scores.size(-1) - 1, -1, -1))  # flip, but without copy
        d_scores = (1 - scores[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * scores
        d_scores[..., -1] = scores[..., -1]

        ret = {}

        if pyramid:
            score_img = einops.rearrange(d_scores, 'n h w m -> n m h w')
            try:
                G = self.generator
            except AttributeError:
                G = self.generator_ema
            ret['scores_pyramid'] = pyramid_resize(score_img, cutoff=G.size_in)

        feature_grid = splat_features_from_scores(ret['scores_pyramid'][size], features, size, channels_last=False)
        ret.update({'feature_grid': feature_grid, 'feature_img': None, 'entropy_img': None})
        if return_metadata:
            metadata = {'xs': xs, 'ys': ys, 'covs': covs, 'raw_scores': scores, 'sizes': sizes,
                        'composed_scores': d_scores, 'features': features}
            ret.update(metadata)
        if viz:
            if viz_score_fn is not None:
                viz_posterior = viz_score_fn(scores)
                scores_viz = (1 - viz_posterior[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * viz_posterior
                scores_viz[..., -1] = viz_posterior[..., -1]
            else:
                scores_viz = d_scores
            ret.update(self.visualize_features(xs, ys, viz_size, features, scores_viz, **kwargs))
        return ret

    def generate_layout(self, noise: Optional[Tensor] = None, return_metadata: bool = False, ema: bool = False,
                        size: Optional[int] = None, viz: bool = False,
                        num_features: Optional[int] = None,
                        metadata: Optional[Dict[str, Tensor]] = None,
                        mlp_idx: Optional[int] = None,
                        score_size: Optional[int] = None,
                        viz_size: Optional[int] = None,
                        truncate: Optional[float] = None,
                        **kwargs) -> Dict[str, Tensor]:
        """
        Args:
            noise: [N x D] tensor of noise
            mlp_idx: idx at which to split layout net MLP used for truncating
            num_features: how many features if not drawn randomly
            ema: use EMA version or not
            size: H, W output for feature grid
            viz: return RGB viz of feature grid
            return_metadata: if true, return an RGB image demonstrating feature placement
            score_size: size at which to render score grid before downsampling to size
            viz_size: visualized grid in RGB dimension
            truncate: if not None, use this as factor for computing truncation. requires self.mean_latent to be set. 0 = no truncation. 1 = full truncation.
            metadata: output in format returned by return_metadata, can be used to generate instead of fwd pass
        Returns: [N x C x H x W] tensor of input, optionally [N x 3 x H_out x W_out] visualization of feature spread
        """
        if num_features is None:
            num_features = random.randint(self.n_features_min, self.n_features_max)
        if metadata is None:
            layout_net = self.layout_net_ema if ema else self.layout_net
            assert noise is not None
            if truncate is not None:
                mlp_idx = -1
                noise = layout_net.mlp[:mlp_idx](noise)
                noise = (self.mean_latent * truncate) + (noise * (1 - truncate))
            metadata = layout_net(noise, num_features, mlp_idx)

        try:
            G = self.generator
        except AttributeError:
            G = self.generator_ema

        ret = self.splat_features(**metadata, size=size or G.size_in, viz_size=viz_size or G.size,
                                  viz=viz, return_metadata=return_metadata, score_size=score_size or (size or G.size),
                                  pyramid=True,
                                  **kwargs)

        if self.spatial_style:
            ret['spatial_style'] = metadata['spatial_style']
        if 'noise' in metadata:
            ret['noise'] = metadata['noise']
        if 'h_stdev' in metadata:
            ret['h_stdev'] = metadata['h_stdev']
        return ret

    def shared_step(self, batch: Tuple[Tensor, dict], batch_idx: int,
                    optimizer_idx: Optional[int] = None, mode: str = 'train') -> Optional[Union[Tensor, dict]]:
        """
        Args:
            batch: tuple of tensor of shape N x C x H x W of images and a dictionary of batch metadata/labels
            batch_idx: pytorch lightning training loop batch index
            optimizer_idx: pytorch lightning optimizer index (0 = G, 1 = D)
            mode:
                `train` returns the total loss and logs losses and images/profiling info.
                `validate`/`test` log total loss and return images
        Returns: see description for `mode` above
        """
        if run_at_step(self.trainer.global_step, self.flush_cache_every_n_steps):
            torch.cuda.empty_cache()
        # Set up modules and data
        train = mode == 'train'
        train_G = train and optimizer_idx == 0 and not self.freeze_G
        train_D = train and (optimizer_idx == 1 or self.freeze_G)
        batch_real, batch_labels = batch
        z = torch.randn(len(batch_real), self.noise_dim).type_as(batch_real)
        info = dict()
        losses = dict()

        log_images = run_at_step(self.trainer.global_step, self.log_images_every_n_steps)
        layout, gen_imgs, latents = self.gen(z, ret_layout=True, ret_latents=True, viz=log_images)
        if latents is not None and not self.spatial_style:
            if latents.ndim == 3:
                latents = latents[0]
            info['latent_norm'] = latents.norm(2, 1).mean()
            info['latent_stdev'] = latents.std(0).mean()

        # Compute various losses
        logits_fake = self.discriminator(gen_imgs)
        if train_G or not train:
            # Log
            losses['G'] = F.softplus(-logits_fake).mean()
            if run_at_step(self.trainer.global_step, self.trainer.log_every_n_steps):
                with torch.no_grad():
                    coords = torch.stack((layout['xs'], layout['ys']), -1)
                    centroids = coords.mean(1, keepdim=True)
                    # only consider spread of elements being used
                    coord_mask = layout['sizes'][:, 1:] > -5
                    info.update({'coord_spread': (coords - centroids)[coord_mask].norm(2, -1).mean()})
                    shift = layout['sizes'][:, 1:]
                    info.update({
                        'shift_mean': shift.mean(),
                        'shift_std': shift.std(-1).mean()
                    })
        if train_D or not train:
            # Discriminate real images
            logits_real = self.discriminator(batch_real)
            # Log
            losses['D_real'] = F.softplus(-logits_real).mean()
            losses['D_fake'] = F.softplus(logits_fake).mean()
            info.update(get_D_stats('fake', logits_fake, gt=False))
            info.update(get_D_stats('real', logits_real, gt=True))

        # Save images
        imgs = {
            'real_imgs': batch_real,
            'gen_imgs': gen_imgs,
            'feature_imgs': layout['feature_img'],
            'entropy_imgs': layout['entropy_img']
        }

        # Compute train regularization loss
        if train_G and run_at_step(batch_idx, self.G_reg_every):
            if self.λ.G_path:
                z = mixing_noise(batch_real, self.dim, self.p_mixing_noise)
                gen_imgs, latents = self.generator(z, return_latents=True)
                losses['G_path'], self.path_len, info['G_path_len'] = G_path_loss(gen_imgs, latents, self.path_len)
                losses['G_path'] = losses['G_path'] * self.G_reg_every
        elif train_D and run_at_step(batch_idx, self.D_reg_every):
            if self.λ.D_R1:
                with autocast(enabled=False):
                    batch_real.requires_grad = True
                    logits_real = self.discriminator(batch_real)
                    R1 = D_R1_loss(logits_real, batch_real)
                    info['D_R1_unscaled'] = R1
                    losses['D_R1'] = R1 * self.D_reg_every

        # Compute final loss and log
        total_loss = f'total_loss_{"G" if train_G else "D"}'
        losses[total_loss] = sum(map(lambda k: losses[k] * self.λ[k], losses))
        isnan = self.alert_nan_loss(losses[total_loss], batch_idx)
        if self.all_gather(isnan).any():
            if self.ipdb_on_nan and is_rank_zero():
                import ipdb
                ipdb.set_trace()
            return
        self.log_scalars(losses, mode)
        self.log_scalars(info, mode)
        # Further logging and terminate
        if mode == "train":
            if train_G:
                if self.accumulate:
                    accumulate(self.generator_ema, self.generator, 0.5 ** (32 / (10 * 1000)))
                    accumulate(self.layout_net_ema, self.layout_net, 0.5 ** (32 / (10 * 1000)))
                if log_images and is_rank_zero():
                    if self.accumulate and self.n_ema_sample:
                        with torch.no_grad():
                            z = self.sample_z.to(self.device)
                            imgs['gen_imgs_ema'] = self.gen(z, ema=True, viz=True)
                            imgs['feature_imgs_ema'] = layout['feature_img']
                    imgs = {k: v.clone().detach().float().cpu() for k, v in imgs.items() if v is not None}
                    self._log_image_dict(imgs, mode, square_grid=False, ncol=len(batch_real))
                if run_at_step(self.trainer.global_step, self.log_fid_every_n_steps) and train_G:
                    self.log_fid(mode)
                self._log_profiler()
            return losses[total_loss]
        else:
            if self.valtest_log_all:
                imgs = self.gather_tensor_dict(imgs)
            return imgs
