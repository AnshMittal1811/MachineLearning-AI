import os
from .base_logger import BaseLogger
import torch


class HistLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = "hist"
        os.makedirs(self.log_path, exist_ok=True)
        self.phase = None
        self.epoch = -1
        self.batch = -1
        self.metric_container = dict()

    def log_batch(self, batch):
        if self.NAME not in batch["output_parser"].keys():
            return
        keys_list = batch["output_parser"][self.NAME]
        if len(keys_list) == 0:
            return
        data = batch["data"]
        self.phase = batch["phase"]
        self.batch = batch["batch"]
        self.epoch = batch["epoch"]
        for k in keys_list:
            if k not in data.keys():
                continue
            if k not in self.metric_container.keys():
                self.metric_container[k] = []
            if isinstance(data[k], torch.Tensor):  # can be a size [B] tensor
                assert (
                    len(data[k].squeeze().shape) <= 1
                ), "Historgram logger only support a shape [B] Tensor or scalar Tensor"
                data[k] = data[k].detach().cpu().numpy().tolist()
            if isinstance(data[k], list):  # can be a list
                self.metric_container[k] += data[k]
            elif isinstance(data[k], int) or isinstance(data[k], float):  # can be a scalar
                self.metric_container.append(float(data[k]))

    def log_phase(self):
        phase = self.phase
        for k, v in self.metric_container.items():
            self.tb.add_histogram("Hist/{}/{}".format(k, phase), torch.Tensor(v), int(self.epoch))
        self.metric_container = dict()