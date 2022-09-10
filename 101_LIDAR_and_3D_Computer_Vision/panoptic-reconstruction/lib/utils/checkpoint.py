# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from pathlib import Path

import torch
from collections import OrderedDict

from lib.modeling.backbone.model_serialization import load_state_dict
from lib.utils.c2_model_loading import load_c2_format
from lib.utils.imports import import_file
from lib.modeling.backbone.model_zoo import cache_url
from lib.utils.logger import logger
from lib.config import config

# TODO review


class Checkpointer:
    def __init__(self, model, optimizer=None, scheduler=None, output_path: Path = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.output_path = output_path

    def save(self, name, **kwargs):
        if not self.output_path:
            return

        data = dict()
        data["model"] = self.model.state_dict()

        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()

        data.update(kwargs)

        save_file = os.path.join(self.output_path, "{}.pth".format(name))
        logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        elif config.MODEL.PRETRAIN:
            f = config.MODEL.PRETRAIN

        if not f:
            # no checkpoint could be found
            logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)

        self._load_model(checkpoint)

        try:
            if "optimizer" in checkpoint and self.optimizer and config.SOLVER.LOAD_OPTIMIZER is True:
                logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            
            if "scheduler" in checkpoint and self.scheduler and config.SOLVER.LOAD_SCHEDULER is True:
                logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

            # return any further checkpoint data
            return checkpoint
        except:
            print('not loading optimizer and scheduler')
            return {}

    def has_checkpoint(self):
        save_file = os.path.join(self.output_path, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):

        save_file = os.path.join(self.output_path, "last_checkpoint")
        save_path = self.output_path

        if not os.path.exists(save_file):
            save_file = os.path.join(config.MODEL.PRETRAIN, "last_checkpoint")
            save_path = config.MODEL.PRETRAIN

        with open(save_file, "r") as f:
            last_saved = f.read()
            last_saved = last_saved.strip()

        if os.path.exists(last_saved) is False and os.path.isabs(last_saved) is False:
            file_name = os.path.basename(last_saved)
            absolute_last_saved = os.path.join(save_path, file_name)

            if os.path.exists(absolute_last_saved):
                last_saved = absolute_last_saved

        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.output_path, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(self, model, optimizer=None, scheduler=None, output_path: Path = None):
        super().__init__(model, optimizer, scheduler, output_path)

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "lib.config.paths_catalog", config.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
