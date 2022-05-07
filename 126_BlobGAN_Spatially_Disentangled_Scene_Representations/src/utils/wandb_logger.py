# From Tim
from __future__ import annotations

__all__ = ["Logger"]

import os
import re
import subprocess
import sys
from ast import literal_eval
from math import sqrt, ceil
from pathlib import Path
from typing import Optional, Union

import omegaconf
import torch
import torchvision.utils as utils
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from wandb.sdk.lib.config_util import ConfigError

from .distributed import is_rank_zero
from .io import yes_or_no
from .misc import recursive_compare


class Logger(WandbLogger):
    def __init__(
            self,
            *,
            name: str,
            project: str,
            entity: str,
            group: Optional[str] = None,
            offline: bool = False,
            log_dir: Optional[str] = './logs',
            **kwargs
    ):
        log_dir = str(Path(log_dir).absolute())

        super().__init__(
            name=name,
            save_dir=log_dir,
            offline=offline,
            project=project,
            log_model=False,
            entity=entity,
            group=group,
            **kwargs
        )

    def log_hyperparams(self, *args, **kwargs):
        pass

    def _file_exists(self, path: str) -> bool:
        try:
            self.experiment.restore(path)
            return True
        except ValueError:
            return False

    def _get_unique_fn(self, filename: str, sep: str = '_') -> str:
        orig_filename, ext = os.path.splitext(filename)
        cfg_ctr = 0
        while self._file_exists(filename):
            cfg_ctr += 1
            filename = f"{orig_filename}{sep}{cfg_ctr}{ext}"
        return filename

    @rank_zero_only
    def save_to_file(self, filename: str, contents: Union[str, bytes], unique_filename: bool = True) -> str:
        if not is_rank_zero():
            return
        if unique_filename:
            filename = self._get_unique_fn(filename)
        self.experiment.save(filename)
        t = type(contents)
        if t is str:
            mode = 'w'
        elif t is bytes:
            mode = 'wb'
        else:
            raise TypeError('Can only save str or bytes')
        (Path(self.experiment.dir) / filename).open(mode).write(contents)
        return filename

    @rank_zero_only
    def log_config(self, config: omegaconf.DictConfig):
        if not is_rank_zero():
            return
        filename = self.save_to_file("hydra_config.yaml", omegaconf.OmegaConf.to_yaml(config))
        params = omegaconf.OmegaConf.to_container(config)
        assert isinstance(params, dict)
        params.pop("wandb", None)

        try:
            self.experiment.config.update(params)
        except ConfigError as e:
            # Config has changed, so confirm with user that this is okay before proceeding
            msg = e.message.split("\n")[0]

            def try_literal_eval(x):
                try:
                    return literal_eval(x)
                except ValueError:
                    return x

            key, old, new = map(try_literal_eval, re.search("key (.*) from (.*) to (.*)", msg).groups())
            print(f'Caution! Parameters have changed!')
            if not (type(old) == type(new) == dict):
                old = {key: old}
                new = {key: new}
            print(recursive_compare(old, new, level=key))
            if yes_or_no('Was this intended?', default=True, timeout=10):
                print(f'Saving new parameters to {filename} and updating W and B config.')
                self.experiment.config.update(params, allow_val_change=True)
            else:
                sys.exit(1)

    @rank_zero_only
    def log_model_summary(self, model: LightningModule):
        if not is_rank_zero():
            return
        self.save_to_file("model_summary.txt", str(ModelSummary(model, max_depth=-1)))

    @torch.no_grad()
    @rank_zero_only
    def log_image_batch(self, name: str, images: Tensor, square_grid: bool = True, commit: bool = False,
                        ncol: Optional[int] = None, **kwargs):
        """
        Args:
            name: Name of key to use for logging
            images: N x C x H x W tensor of images
            square_grid: whether to render images into a square grid
            commit: whether to commit log to wandb or not
            ncol: analogous to nrow in make_grid, control how many images are in each column
            **kwargs: passed onto make_grid
        """
        if not is_rank_zero():
            return
        assert not (square_grid and ncol is not None), "Set either square_grid or ncol"
        if square_grid:
            kwargs['nrow'] = ceil(sqrt(len(images)))
        elif ncol is not None:
            kwargs['nrow'] = ceil(len(images) / ncol)
        image_grid = utils.make_grid(
            images.float(), normalize=True, value_range=(-1, 1), **kwargs
        )
        wandb_image = wandb.Image(image_grid.float().cpu())
        self.experiment.log({name: wandb_image}, commit=commit)

    @rank_zero_only
    def log_code(self):
        if not is_rank_zero():
            return
        codetar = subprocess.run(
            ['tar', '--exclude=*.pyc', '--exclude=__pycache__', '--exclude=*.pt','--exclude=*.pkl', '-cvJf', '-', 'src'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        self.save_to_file('code.tar.xz', codetar)
