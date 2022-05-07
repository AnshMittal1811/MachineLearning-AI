from __future__ import annotations

__all__ = ["GAN"]

from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from cleanfid import fid
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.optim import Optimizer

from models import networks
from models.base import BaseModule
from utils import FromConfig, run_at_step, get_D_stats, G_path_loss, D_R1_loss, freeze, is_rank_zero, accumulate, \
    mixing_noise


@dataclass
class Lossλs:
    D_real: float = 1
    D_fake: float = 1
    D_R1: float = 5
    G: float = 1
    G_path: float = 2

    def __getitem__(self, key):
        return super().__getattribute__(key)


@dataclass(eq=False)
class GAN(BaseModule):
    # Modules
    generator: FromConfig[nn.Module]
    discriminator: FromConfig[nn.Module]
    # Module parameters
    dim: int = 256
    resolution: int = 128
    p_mixing_noise: float = 0.9
    n_ema_sample: int = 16
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
    log_fid_every_epoch: bool = True
    fid_n_imgs: Optional[int] = 5000
    fid_num_workers: Optional[int] = 24
    valtest_log_all: bool = False
    accumulate: bool = True

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.discriminator = networks.get_network(**self.discriminator)
        self.generator_ema = networks.get_network(**self.generator)
        self.generator = networks.get_network(**self.generator)
        if self.freeze_G:
            self.generator.eval()
            freeze(self.generator)
        if self.accumulate:
            self.generator_ema.eval()
            freeze(self.generator_ema)
            accumulate(self.generator_ema, self.generator, 0)
        else:
            del self.generator_ema
        self.λ = Lossλs(**self.λ)
        self.register_buffer('sample_z', torch.randn(self.n_ema_sample, self.dim))
        # self.sample_z = torch.randn(self.n_ema_sample, self.dim)

    # Initialization and state management
    def on_train_start(self):
        super().on_train_start()
        # Validate parameters w.r.t. trainer (must be done here since trainer is not attached as property yet in init)
        assert self.log_images_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_images_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder'
        if self.log_timing_every_n_steps > -1:
            assert self.log_timing_every_n_steps % self.trainer.log_every_n_steps == 0, \
                '`model.log_images_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder'
        assert self.log_fid_every_n_steps < 0 or self.log_fid_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_fid_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder'

    def configure_optimizers(self) -> Union[optim, List[optim]]:
        G_reg_ratio = self.G_reg_every / ((self.G_reg_every + 1) or -1)
        D_reg_ratio = self.D_reg_every / ((self.D_reg_every + 1) or -1)
        _requires_grad = lambda p: p.requires_grad
        G_optim = torch.optim.Adam(filter(_requires_grad, self.generator.parameters()), lr=self.lr * G_reg_ratio,
                                   betas=(0 ** G_reg_ratio, 0.99 ** G_reg_ratio), eps=self.eps)
        D_optim = torch.optim.Adam(filter(_requires_grad, self.discriminator.parameters()), lr=self.lr * D_reg_ratio,
                                   betas=(0 ** D_reg_ratio, 0.99 ** D_reg_ratio), eps=self.eps)
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
        optimizer.step(closure=optimizer_closure)

    def training_epoch_end(self, *args, **kwargs):
        if self.log_fid_every_epoch:
            try:
                self.log_fid("train")
            except:
                pass

    @torch.no_grad()
    def log_fid(self, mode, **kwargs):
        def gen_fn(z):
            if self.accumulate:
                out = self.generator_ema([z], return_image_only=True).add_(1).div_(2).mul_(255)
            else:
                out = self.generator([z], return_image_only=True).add_(1).div_(2).mul_(255)
            return out
        if is_rank_zero():
            dataset = self.trainer.datamodule.path.name
            fid_split = "train" if dataset == "church" else "custom"
            fid_score = fid.compute_fid(gen=gen_fn, dataset_name=f"lsun_{dataset}",
                                        dataset_res=256, num_gen=self.fid_n_imgs,
                                        dataset_split=fid_split, device=self.device,
                                        num_workers=self.fid_num_workers)
        else:
            fid_score = 0.0
        fid_score = self.all_gather(fid_score).max().item()
        self.log_scalars({'fid': fid_score}, mode, **kwargs)

    # Training and evaluation
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
         # Set up modules and data
        train = mode == 'train'
        train_G = train and optimizer_idx == 0 and not self.freeze_G
        train_D = train and (optimizer_idx == 1 or self.freeze_G)
        batch_real, batch_labels = batch
        # z = torch.randn(len(batch_real), self.dim).type_as(batch_real)
        info = dict()
        losses = dict()
        z = mixing_noise(batch_real, self.dim, self.p_mixing_noise)

        gen_imgs, latents = self.generator(z, return_latents=True)

        if latents is not None:
            if latents.ndim == 3:
                latents = latents[:, 0]
            info['latent_norm'] = latents.norm(2, 1).mean()
            info['latent_stdev'] = latents.std(0).mean()

        # Compute various losses
        logits_fake = self.discriminator(gen_imgs)
        if train_G or not train:
            # Log
            losses['G'] = F.softplus(-logits_fake).mean()
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
        }
        imgs = {k: v.clone().detach().float().cpu() for k, v in imgs.items()}

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
        losses['total_loss'] = sum(map(lambda k: losses[k] * self.λ[k], losses))
        # if losses['total_loss'] > 20 and is_rank_zero():
        #     import ipdb
        #     ipdb.set_trace()
        if self.alert_nan_loss(losses['total_loss'], batch_idx):
            if is_rank_zero():
                import ipdb
                ipdb.set_trace()
            return
        self.log_scalars(losses, mode)
        self.log_scalars(info, mode)
        # Further logging and terminate
        if mode == "train":
            if train_G and self.accumulate:
                accumulate(self.generator_ema, self.generator, 0.5 ** (32 / (10 * 1000)))
            if run_at_step(self.trainer.global_step, self.log_images_every_n_steps):
                if self.accumulate:
                    with torch.no_grad():
                        imgs['gen_imgs_ema'], _ = self.generator_ema([self.sample_z])
                self._log_image_dict(imgs, mode, square_grid=False, ncol=len(batch_real))
            if run_at_step(self.trainer.global_step, self.log_fid_every_n_steps) and is_rank_zero() and train_G:
                self.log_fid(mode)
            self._log_profiler()
            return losses['total_loss']
        else:
            if self.valtest_log_all:
                imgs = self.gather_tensor_dict(imgs)
            return imgs
