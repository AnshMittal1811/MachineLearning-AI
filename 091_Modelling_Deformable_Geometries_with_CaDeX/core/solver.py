"""
solve the optimization on a finite dataset
"""
from copy import deepcopy
import os
import logging
import torch
from torch.utils.data import DataLoader
import gc


class Solver(object):
    def __init__(self, cfg, model, datasets_dict, logger):
        self.cfg = deepcopy(cfg)

        self.modes = self.cfg["modes"]
        self.dataloader_dict = {}
        for mode in cfg["modes"]:  # prepare dataloader
            if mode.lower() == "train" or cfg["evaluation"]["batch_size"] < 0:
                bs = cfg["training"]["batch_size"]
            else:
                bs = cfg["evaluation"]["batch_size"]
            # if mode.lower() == "train":
            #     n_workers = cfg["dataset"]["num_workers"]
            # else:
            #     n_workers = min(4, cfg["dataset"]["num_workers"])
            n_workers = cfg["dataset"]["num_workers"]
            # decide shuffle
            shuffle_dataset = True if mode in ["train"] else False
            if "shuffle" in cfg["evaluation"].keys():
                if cfg["evaluation"]["shuffle"]:
                    shuffle_dataset = True
            logging.debug(f"{mode} dataloader use pin_mem={cfg['dataset']['pin_mem']}")
            self.dataloader_dict[mode] = DataLoader(
                datasets_dict[mode],
                batch_size=bs,
                shuffle=shuffle_dataset,
                num_workers=n_workers,
                pin_memory=cfg["dataset"]["pin_mem"],
                drop_last=mode == "train",  # ! check this
            )
        self.model = model
        self.logger = logger

        self.current_epoch = 1
        self.batch_count = 0
        self.batch_in_epoch_count = 0
        self.total_epoch = cfg["training"]["total_epoch"]

        self.eval_every_epoch = int(cfg["evaluation"]["eval_every_epoch"])

        self.clear_phase_cache = cfg["training"]["clear_phase_cache"]

        # save lr decay
        self.lr_config = self.init_lr_schedule()

        # handle resume and initialization
        if cfg["resume"]:  # resume > initialization
            self.solver_resume()
        elif len(cfg["training"]["initialize_network_file"]) > 0:
            assert isinstance(
                cfg["training"]["initialize_network_file"], list
            ), "Initialization from file config should be a list fo file path"
            self.initialize_from_file(
                cfg["training"]["initialize_network_file"],
                cfg["training"]["initialize_network_name"],
            )
        self.model.to_gpus()

        # control viz in model and logger
        log_config = self.cfg["logging"]
        self.viz_interval_epoch = log_config["viz_epoch_interval"]
        self.viz_interval_train_batch = log_config["viz_training_batch_interval"]
        self.viz_interval_nontrain_batch = log_config["viz_nontrain_batch_interval"]
        self.viz_flag = False

        return

    def solver_resume(self):
        resume_key = self.cfg["resume"]
        checkpoint_dir = os.path.join(
            self.cfg["root"], "log", self.cfg["logging"]["log_dir"], "checkpoint"
        )
        if resume_key == "latest":
            checkpoint_founded = os.listdir(checkpoint_dir)
            checkpoint_fn = None
            for fn in checkpoint_founded:
                if fn.endswith("_latest.pt"):
                    checkpoint_fn = os.path.join(checkpoint_dir, fn)
        else:
            checkpoint_fn = os.path.join(checkpoint_dir, resume_key + ".pt")
        checkpoint = torch.load(checkpoint_fn)
        logging.info("Checkpoint {} Loaded".format(checkpoint_fn))
        self.current_epoch = checkpoint["epoch"]
        self.batch_count = checkpoint["batch"]
        self.model.model_resume(checkpoint, is_initialization=False)
        self.adjust_lr()
        self.current_epoch += 1
        return

    def initialize_from_file(self, filelist, network_name):
        for fn in filelist:
            checkpoint = torch.load(fn)
            logging.info("Initialization {} Loaded".format(fn))
            self.model.model_resume(checkpoint, is_initialization=True, network_name=network_name)
        return

    def run(self):
        logging.info("Start Running...")
        while self.current_epoch <= self.total_epoch:
            for mode in self.modes:
                if self.clear_phase_cache:
                    torch.cuda.empty_cache()
                if mode.lower() != "train" and self.current_epoch % self.eval_every_epoch != 0:
                    continue  # for val and test, skip if not meets eval epoch interval
                batch_total_num = len(self.dataloader_dict[mode])
                self.batch_in_epoch_count = 0
                for batch in iter(self.dataloader_dict[mode]):
                    self.batch_in_epoch_count += 1
                    self.batch_count += 1
                    self.viz_flag = self.viz_state(mode)
                    batch[0]["epoch"] = self.current_epoch
                    if mode == "train":
                        batch = self.model.train_batch(batch, self.viz_flag)
                    else:
                        batch = self.model.val_batch(batch, self.viz_flag)
                    batch = self.wrap_output(batch, batch_total_num, mode=mode)
                    self.logger.log_batch(batch)
                self.logger.log_phase()
                gc.collect()
            self.adjust_lr()
            self.current_epoch += 1
        self.logger.end_log()
        return

    def wrap_output(self, batch, batch_total, mode="train"):
        assert "meta_info" in batch.keys()
        wrapped = dict()

        wrapped["viz_flag"] = self.viz_flag

        wrapped["batch"] = self.batch_count
        wrapped["batch_in_epoch"] = self.batch_in_epoch_count
        wrapped["batch_total"] = batch_total
        wrapped["epoch"] = self.current_epoch
        wrapped["phase"] = mode.lower()

        wrapped["output_parser"] = self.model.output_specs
        wrapped["save_method"] = self.model.save_checkpoint
        wrapped["meta_info"] = batch["meta_info"]
        wrapped["data"] = batch

        return wrapped

    def init_lr_schedule(self):
        schedule = self.cfg["training"]["optim"]
        schedule_keys = schedule.keys()
        if "all" in schedule_keys:
            schedule_keys = ["all"]
        for k in schedule_keys:
            if isinstance(schedule[k]["decay_schedule"], int):
                assert isinstance(schedule[k]["decay_factor"], float)
                schedule[k]["decay_schedule"] = [
                    i
                    for i in range(
                        schedule[k]["decay_schedule"],
                        self.total_epoch,
                        schedule[k]["decay_schedule"],
                    )
                ]
                schedule[k]["decay_factor"] = [schedule[k]["decay_factor"]] * len(
                    schedule[k]["decay_schedule"]
                )
            elif isinstance(schedule[k]["decay_schedule"], list):
                assert isinstance(schedule[k]["decay_factor"], list)
            else:
                assert RuntimeError("Lr Schedule error!")
        return schedule

    def adjust_lr(self, specific_epoch=None):
        epoch = self.current_epoch if specific_epoch is None else specific_epoch
        for k in self.lr_config.keys():
            if epoch in self.lr_config[k]["decay_schedule"]:
                optimizer = self.model.optimizer_dict[k]
                for param_group in optimizer.param_groups:
                    lr_before = param_group["lr"]
                    factor = self.lr_config[k]["decay_factor"][
                        self.lr_config[k]["decay_schedule"].index(epoch)
                    ]
                    param_group["lr"] = param_group["lr"] * factor
                    param_group["lr"] = max(param_group["lr"], self.lr_config[k]["lr_min"])
                    lr_new = param_group["lr"]
                logging.info(
                    "After epoch{}, Change {} lr {:.5f} to {:.5f}".format(
                        epoch, k, lr_before, lr_new
                    )
                )

    def viz_state(self, mode):
        viz_flag = True
        if self.current_epoch % self.viz_interval_epoch != 0:
            viz_flag = False
        if mode == "train":
            if self.batch_in_epoch_count % self.viz_interval_train_batch != 0:
                viz_flag = False
        else:
            if self.batch_in_epoch_count % self.viz_interval_nontrain_batch != 0:
                viz_flag = False
        return viz_flag
