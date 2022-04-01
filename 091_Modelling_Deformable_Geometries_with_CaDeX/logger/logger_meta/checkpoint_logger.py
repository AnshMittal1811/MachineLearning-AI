import os
import shutil
from .base_logger import BaseLogger
import numpy as np
import torch
import logging


class CheckpointLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = "checkpoint"
        os.makedirs(self.log_path, exist_ok=True)
        self.phase = "train"
        self.current_epoch = -1
        self.current_batch = -1
        save_interval = cfg["logging"]["checkpoint_epoch"]
        if isinstance(save_interval, int):
            self.save_epoch_list = [i for i in range(0, self.total_epoch + 1, save_interval)]
        elif isinstance(save_interval, list):
            self.save_epoch_list = save_interval
        else:
            raise RuntimeError("Checkpoint saving init invalid!")
        self.save_method = None
        # use as model select
        self.model_select_metric = cfg["logging"]["model_select_metric"]
        self.model_select_larger = cfg["logging"]["model_select_larger"]
        self.model_select_best = -np.inf if self.model_select_larger else np.inf
        self.model_select_buffer = []

    def log_batch(self, batch):
        self.phase = batch["phase"]
        self.current_epoch = batch["epoch"]
        self.current_batch = batch["batch"]
        if self.save_method is None:
            self.save_method = batch["save_method"]
        # update model selection metric
        if self.phase.startswith("val"):  # val or vali
            if self.model_select_metric in batch["data"].keys():
                metric = batch["data"][self.model_select_metric]
                if isinstance(metric, torch.Tensor):
                    metric = metric.detach().cpu()
                self.model_select_buffer.append(float(metric))

    def log_phase(self):
        batch_epoch_info = {"batch": self.current_batch, "epoch": self.current_epoch}
        if self.phase == "train" and (self.current_epoch in self.save_epoch_list):  # log the trace
            self.save_method(
                os.path.join(self.log_path, "%d.pt" % self.current_epoch), batch_epoch_info
            )
        if self.phase == "train":  # log the latest
            if any(
                [True if fn.endswith("latest.pt") else False for fn in os.listdir(self.log_path)]
            ):
                os.system("rm " + os.path.join(self.log_path, "*_latest.pt"))
            self.save_method(
                os.path.join(self.log_path, "%d_latest.pt" % self.current_epoch), batch_epoch_info
            )
        if self.phase.startswith("val"):  # model selection
            if len(self.model_select_buffer) > 0:
                # model select
                mean_metric = np.array(self.model_select_buffer).mean()
                select = self.better(old=self.model_select_best, new=mean_metric)
                if select:
                    # if there exist a previous best model, double check it!
                    old_fn = self.find_selected()
                    if old_fn is not None:
                        old_fn = os.path.join(self.log_path, old_fn)
                        old_metric = torch.load(old_fn)["select_metric"]
                        select = self.better(old=old_metric, new=mean_metric)
                        if select:  # remove old selection
                            os.system("rm " + old_fn)
                    if select:  # if still select
                        fn = os.path.join(
                            self.log_path, "selected.pt"
                        )
                        batch_epoch_info["select_metric"] = mean_metric
                        self.save_method(fn, batch_epoch_info)
                        logging.info("Select epoch {} model".format(self.current_epoch))
            self.model_select_buffer = []

    def better(self, old, new):
        select = False
        if self.model_select_larger and new > old:
            select = True
        if (not self.model_select_larger) and new < old:
            select = True
        return select

    def find_selected(self):
        ckpts = os.listdir(self.log_path)
        found = None
        for ck in ckpts:
            if ck.endswith("selected.pt"):
                found = ck
                break
        return found
