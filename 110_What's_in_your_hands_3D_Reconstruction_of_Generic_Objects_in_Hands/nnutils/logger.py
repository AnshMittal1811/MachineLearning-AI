# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import logging
import os
import torchvision.utils as vutils
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger

from nnutils import image_utils

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")


class MyLogger(TensorBoardLogger):
    @staticmethod
    def loss_str(losses: dict):
        """Dict: {key:(N, )}
        :return [str, ] len of N """
        N = list(losses.values())[0].size(0)
        loss_list = [''] * N
        for key, value in losses.items():
            for n in range(N):
                loss_list[n] += '%s: %.4f\n' % (key, value[n])
        return loss_list

    def __init__(self, resume=True, subfolder='', **kwargs):
        self.subfolder = subfolder if subfolder is not None and len(subfolder) > 0 else 'train'
        if self.subfolder != 'train':
            kwargs['version'] = os.path.join(kwargs.get('version'), self.subfolder)
        super(MyLogger, self).__init__(**kwargs)
        if not resume:
            logging.warn('REMOVE %s' % self.log_dir)
            cmd = 'rm -rf %s' % self.log_dir 
            print(cmd)
            # cmd = 'rm -rf {0}/checkpoints {0}/train {0}/hparams.yaml {0}/event*'.format(self.log_dir)
            # log.info('overwrite  %s' % cmd)
            os.system(cmd)        
        log_file = "{0}/{1}.log".format(self.log_dir, self.subfolder)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        fileHandler = logging.FileHandler(log_file, mode='a')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        log.addHandler(fileHandler)
        self.log = log

    @property
    def local_dir(self) -> str:
        """
        directory to save training images.
        """
        # create a pseudo standard path ala test-tube
        if self.subfolder == 'train':
            local_dir = os.path.join(self.log_dir, self.subfolder)
        else:
            local_dir = self.log_dir
        self._fs.makedirs(local_dir, exist_ok=True)
        return local_dir

    @property
    def model_name(self) -> str:
        """
        model signature: name/version
        """
        return os.path.join(self.name, str(self.version))

    @rank_zero_only
    def add_images(self, iteration, images, name=''):
        """
        :param iteration:
        :param images:  Tensor (N, C, H, W), in range (-1, 1)
        :param name:
        :return:
        """
        images = images.cpu().detach()
        x = vutils.make_grid(images)
        # self.experiment.add_image(name, x / 2 + 0.5, iteration)
        self.experiment.add_image(name, x, iteration)

    @rank_zero_only
    def add_losses(self, t, losses, name='', ):
        for k, v in losses.items():
            index = name + '/%s' % k
            self.experiment.add_scalar(index, v, t)

    @rank_zero_only
    def print(self, t, epoch, losses, total_loss):
        log = self.log
        log.info('[Epoch %2d] iter: %d of model %s' % (epoch, t, self.model_name))
        log.info('\tTotal Loss: %.6f' % total_loss)
        for k in losses:
            log.info('\t\t%s: %.6f' % (k, losses[k]))

    @rank_zero_only
    def save_images(self, it, images, name, scale=True, merge=1, bg=None, mask=None, r=0.75, mode='rgb', **kwargs):
        """
        :param it:
        :param images:
        :param name:
        :param scale: if RGB is in [-1, 1]
        :return:
        """
        fname = os.path.join(self.local_dir, '%d_%s' % (it, name))
        if mode == 'rgb':
            image_utils.save_images(images, fname, merge=merge, scale=scale, bg=bg, mask=mask, r=r, **kwargs)
        elif mode == 'heat':
            image_utils.save_heatmap(images, fname, merge=merge, scale=scale, **kwargs)

    @rank_zero_only
    def save_gif(self, it, image_list, name, scale=False, merge=1, **kwargs):
        if len(image_list) == 0:
            log.info('not save empty gif')
            return
        fname = os.path.join(self.local_dir, '%d_%s' % (it, name))
        image_utils.save_gif(image_list, fname, merge=merge, scale=scale, **kwargs)
