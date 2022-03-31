"""
Training scheduler.
Allows to modify losses and parameters as the training goes.
"""

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modules_sdf import SDFIBRNet


class Scheduler(object):
    """
    Training scheduler.
    Allows to modify losses and parameters,.. as the training goes.
    """

    def __init__(self, opt, model: SDFIBRNet, optimizer, writer: SummaryWriter):
        self.opt = opt
        self.model = model
        self.optimizer = optimizer
        self.writer = writer

        # Backup initial params.
        self.opt.rt_mask_alpha_0 = self.opt.rt_mask_alpha
        self.opt.base_lr = self.opt.lr
        self.opt.base_lr_sdf = self.opt.lr_sdf
        self.opt.lr_sdf = self.opt.base_lr_sdf
        self.opt.lr_color = self.opt.base_lr_sdf if getattr(self.opt, "lr_color", 0) else 0

        self.opt.lr_image_enc = self.opt.base_lr
        self.opt.lr_image_dec = self.opt.base_lr
        self.opt.lr_agg_net = self.opt.base_lr

        self.opt.base_meta_lr = self.opt.meta_lr if getattr(self.opt, "meta_lr", 0) else 0
        self.opt.meta_lr_decay_steps = self.opt.meta_lr_decay_steps if getattr(self.opt, "meta_lr_decay_steps", 0) else 0

        if opt.train_decoder_sdf:
            self.set_lr('sdf', self.opt.lr_sdf, 0)
            self.set_lr('color', self.opt.lr_color, 0)
        if opt.train_image_encoder:
            self.set_lr('image_enc', self.opt.lr_image_enc, 0)
        if opt.train_feature_decoder:
            self.set_lr('image_dec', self.opt.lr_image_dec, 0)

    def update(self, epoch: int, steps: int, loss: torch.Tensor):
        """
        Call after each training step.
        """
        # Update rt_mask_alpha.
        alpha_periods = 0
        if self.opt.rt_mask_alpha_period_epochs > 0:
            # Epoch drive.
            alpha_periods = epoch // self.opt.rt_mask_alpha_period_epochs
        elif self.opt.rt_mask_alpha_period > 0:
            # Step drive.
            alpha_periods = steps // self.opt.rt_mask_alpha_period

        # Clamp max periods.
        if self.opt.rt_mask_alpha_period_max > -1:
            alpha_periods = np.minimum(alpha_periods, self.opt.rt_mask_alpha_period_max)

        # Update parameter.
        self.opt.rt_mask_alpha = self.opt.rt_mask_alpha_0 * 2**alpha_periods
        self.writer.add_scalar("rt_mask_alpha", self.opt.rt_mask_alpha, steps)

        # Update learning rates.

        # Decay
        if self.opt.lr_sdf_decay_steps > 0:
            interval = steps // self.opt.lr_sdf_decay_steps
            interval = self.process_interval(interval)
            self.opt.lr_sdf = self.opt.base_lr_sdf * self.opt.lr_decay_factor**interval
        else:
            self.opt.lr_sdf = self.opt.base_lr_sdf

        if self.opt.lr_encdec_decay_steps > 0:
            interval = steps // self.opt.lr_encdec_decay_steps
            interval = self.process_interval(interval)
            self.opt.lr_image_enc = self.opt.base_lr * self.opt.lr_decay_factor ** interval
            self.opt.lr_image_dec = self.opt.base_lr * self.opt.lr_decay_factor ** interval
            self.opt.lr_agg_net = self.opt.base_lr * self.opt.lr_decay_factor ** interval
        else:
            self.opt.lr_image_enc = self.opt.base_lr
            self.opt.lr_image_dec = self.opt.base_lr
            self.opt.lr_agg_net = self.opt.base_lr

        if getattr(self.opt, "lr_color_decay_steps", 0) and self.opt.lr_color_decay_steps > 0:
            interval = steps // self.opt.lr_color_decay_steps
            self.opt.lr_color = self.opt.base_lr_sdf * self.opt.lr_decay_factor**interval
        else:
            self.opt.lr_color = self.opt.base_lr_sdf

        if self.opt.meta_lr_decay_steps > 0:
            interval = steps // self.opt.meta_lr_decay_steps
            self.opt.meta_lr = self.opt.base_meta_lr * self.opt.meta_lr_decay_factor**interval
        else:
            self.opt.meta_lr = self.opt.base_meta_lr

        # Alternating.
        if self.opt.lr_alternating_interval > 0:
            interval = steps // self.opt.lr_alternating_interval
            if interval % 2 == 0:
                # Start with shape.
                self.opt.lr_color = 0
            else:
                # Start with shape.
                self.opt.lr_sdf = 0

        self.set_lr('sdf', self.opt.lr_sdf, steps)
        self.set_lr('color', self.opt.lr_color, steps)
        self.set_lr('image_enc', self.opt.lr_image_enc, steps)
        self.set_lr('image_dec', self.opt.lr_image_dec, steps)
        self.set_lr('agg', self.opt.lr_agg_net, steps)

    def set_lr(self, parameter_name, lr, steps):
        """
        Adjusts optimizer.
        """
        group = [p for p in self.optimizer.param_groups if p['name'] == parameter_name]
        if group:
            group[0]['lr'] = lr
        self.writer.add_scalar(f"lr_{parameter_name}", lr, steps)

    def process_interval(self, interval):
        if interval <= 1:
            return interval
        elif 1 < interval <= 2:
            return 1
        elif 2 < interval <= 6:
            return 2
        elif 6 < interval <= 14:
            return 3
        elif 14 < interval < 30:
            return 4
        elif 30 < interval < 62:
            return 5
        else:
            return 6
