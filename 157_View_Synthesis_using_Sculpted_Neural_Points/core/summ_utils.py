import torch
import cv2
import numpy as np

import sys
sys.path.append('core')
import frame_utils

from torch.utils.tensorboard import SummaryWriter



class Logger:
    def __init__(self, model, scheduler, output, SUM_FREQ, img_log_freq=100, tb_logdir=None):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=tb_logdir)
        self.output = output
        self.SUM_FREQ = SUM_FREQ
        self.img_log_freq = img_log_freq

    def set_global_step(self, global_step):
        self.total_steps = global_step

    def _print_training_status(self):
        SUM_FREQ = self.SUM_FREQ
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        # training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        training_str = "[{:6d}]".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)
        if not self.output is None:
            f = open(self.output, "a")
            f.write(f"{training_str + metrics_str}\n")
            f.close()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, prefix=None):
        SUM_FREQ = self.SUM_FREQ

        for (key, v) in metrics.items():
            if prefix is not None:
                key = '%s/%s' % (prefix, key)

            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += v

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results, prefix=None):
        for (key, v) in results.items():
            if prefix is not None:
                key = '%s/%s' % (prefix, key)

            self.writer.add_scalar(key, v, self.total_steps)

    def summ_rgb(self, tag, rgb, mask=None, bgr2rgb=True, force_save=False):
        # rgb should have shape B x 3 x H x W, and be in range [-0.5, 0.5]
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            rgb = (rgb + 1.0) / 2.0
            rgb = torch.clamp(rgb, 0.0, 1.0)
            if bgr2rgb:
                rgb = rgb[:, [2, 1, 0]]
            if mask is not None:
                rgb = rgb * mask + 1.0 * (1.0 - mask)  # make bkg white
            # if len(rgb) > 1:
            #     for i in range(len(rgb)):
            #         self.writer.add_image(tag + ('_b%d'%i), rgb[i], self.total_steps)
            # else:
            #     self.writer.add_image(tag, rgb[0], self.total_steps)
            self.writer.add_image(tag, rgb[0], self.total_steps)
            return True
        else:
            return False

    def summ_rgbs(self, tag, rgbs, fps=10, bgr2rgb=True, force_save=False):
        # rgbs should have shape N x 3 x H x W, and be in range [-1, 1]
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            rgbs = (rgbs + 1.0) / 2.0
            rgbs = torch.clamp(rgbs, 0.0, 1.0)
            if bgr2rgb:
                rgbs = rgbs[:, [2, 1, 0]]
            self.writer.add_video(tag, rgbs.unsqueeze(0), self.total_steps, fps=fps)
            return True
        else:
            return False

    def summ_oned(self, tag, img, force_save=False):
        # make sure the img has range [0,1]
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            img = torch.clamp(img, 0.0, 1.0)
            self.writer.add_image(tag, img[0, 0], self.total_steps, dataformats='HW')
            return True
        else:
            return False

    def summ_diff(self, tag, im1, im2, vmin=0, vmax=100, force_save=False):
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            im1 = (im1[0]).permute(1, 2, 0).cpu().numpy() # H x W x 3
            im2 = (im2[0]).permute(1, 2, 0).cpu().numpy()

            im1 = (im1 + 1.0) / 2.0 * 255.0
            im2 = (im2 + 1.0) / 2.0 * 255.0
            vis = frame_utils.grayscale_visualization(np.mean(np.abs((im1 - im2)), axis=2), 'L1 diff', vmin=vmin, vmax=vmax)

            self.writer.add_image(tag, vis, self.total_steps, dataformats='HWC')
            return True

        else:
            return False

    def summ_hist(self, tag, tensor, force_save=False):
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            self.writer.add_histogram(tag, tensor, self.total_steps)
            return True
        else:
            return False

    def close(self):
        self.writer.close()