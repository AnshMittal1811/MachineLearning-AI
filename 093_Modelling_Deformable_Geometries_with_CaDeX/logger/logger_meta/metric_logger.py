import matplotlib
import os
import shutil
from .base_logger import BaseLogger
import time
import logging
import torch
from pprint import pformat

matplotlib.use("Agg")


class MetricLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = "metric"
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(os.path.join(self.log_path, "batchwise"), exist_ok=True)
        self.phase = None
        self.epoch = -1
        self.batch = -1
        self.batch_in_epoch = -1
        self.metric_container = dict()
        self.phase_time_start = time.time()
        self.print_time = time.time()

    def log_batch(self, batch):
        """
        - add each metric to tensorboard
        - record each metric for epoch save
        - display in terminal, displayed metric is averaged
        """
        if self.NAME not in batch["output_parser"].keys():
            return
        keys_list = batch["output_parser"][self.NAME]
        if len(keys_list) == 0:
            return
        data = batch["data"]
        total_batch = batch["batch_total"]
        self.phase = batch["phase"]
        self.batch = batch["batch"]
        self.batch_in_epoch = batch["batch_in_epoch"]
        self.epoch = batch["epoch"]
        print_dict = {}
        for k in keys_list:
            if k not in data.keys():
                continue
            if k not in self.metric_container.keys():
                self.metric_container[k] = [data[k]]
            else:
                self.metric_container[k].append(data[k])
            self.tb.add_scalars("Metric-BatchWise/" + k, {self.phase: float(data[k])}, self.batch)
            print_dict[k] = float(data[k])

        if time.time() - self.print_time > 2:
            batch_size = self.batch_size if self.phase.lower() == "train" else self.eval_batch_size
            time_spent = (time.time() - self.phase_time_start) / 60
            time_total = time_spent / (self.batch_in_epoch + 1e-6) * total_batch
            logging.info(
                "{} | Epoch {}/{} | Steps {}/{} | Time {:.3f}min/{:.3f}min ".format(
                    self.phase,
                    self.epoch,
                    self.total_epoch,
                    self.batch_in_epoch * batch_size,
                    total_batch * batch_size,
                    time_spent,
                    time_total,
                )
            )
            logging.info("Metric:\n{}".format(pformat(print_dict, indent=2, compact=True)))
            logging.info("." * 80)
            self.print_time = time.time()

    def log_phase(self):
        """
        - save the batch-wise scalar in file
        - save corresponding figure for each scalar
        For epoch wise metric
        - add to tb
        """
        phase = self.phase
        for k, v in self.metric_container.items():
            # add average to tb epoch wise
            mean = sum(v) / len(v)
            self.tb.add_scalars("Metric-EpochWise/" + k, {phase: float(mean)}, int(self.epoch))
            self.tb.add_histogram(
                "Metric-EpochWise/{}/{}".format(phase, k), torch.Tensor(v), int(self.epoch)
            )
        logging.debug(
            "Finish Epoch {} Phase {} in {}min".format(
                self.epoch, self.phase, (time.time() - self.phase_time_start) / 60.0
            )
        )
        print("\n" + "=" * shutil.get_terminal_size()[0])
        self.metric_container = dict()
        self.phase_time_start = time.time()