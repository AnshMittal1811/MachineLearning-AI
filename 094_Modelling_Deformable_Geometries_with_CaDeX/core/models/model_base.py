import torch.nn as nn
import torch
import copy
import logging
import numpy as np


class ModelBase(object):
    def __init__(self, cfg, network):
        """
        Model Base
        """
        self.cfg = copy.deepcopy(cfg)
        self.__dataparallel_flag__ = False
        self.network = network
        self.optimizer_specs = self.cfg["training"]["optim"]
        self.optimizer_dict = self._register_optimizer()
        # self.to_gpus()
        self.output_specs = {
            "metric": [],
        }
        self.grad_clip = float(cfg["training"]["grad_clip"])
        self.loss_clip = float(cfg["training"]["loss_clip"])
        return

    def _register_optimizer(self):
        optimizer_dict = {}
        parameter_keys = self.optimizer_specs.keys()
        logging.debug("Config defines {} network parameters optimization".format(parameter_keys))
        # if len(parameter_keys) != len(self.network.network_dict.keys()):
        #     logging.warning("Network Components != Optimizer Config")
        if "all" in parameter_keys:
            optimizer = torch.optim.Adam(
                params=self.network.parameters(),
                lr=self.optimizer_specs["all"]["lr"],
            )
            optimizer_dict["all"] = optimizer
        else:
            for key in parameter_keys:
                try:
                    optimizer = torch.optim.Adam(
                        params=self.network.network_dict[key].parameters(),
                        lr=self.optimizer_specs[key]["lr"],
                    )
                    optimizer_dict[key] = optimizer
                except:
                    raise RuntimeError(
                        "Optimizer registration of network component {} fail!".format(key)
                    )
        return optimizer_dict

    def count_parameters(self):
        net = (
            self.network.module.network_dict
            if self.__dataparallel_flag__
            else self.network.network_dict
        )
        for k, v in net.items():
            count = sum(p.numel() for p in v.parameters())
            logging.info("Model-{} has {} parameters".format(k, count))

    def _preprocess(self, batch, viz_flag=False):
        """
        Additional operation if necessary before send batch to network
        """
        data, meta_info = batch
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda().float()
        data["phase"] = meta_info["mode"][0]
        data["viz_flag"] = viz_flag
        batch = {"model_input": data, "meta_info": meta_info}
        return batch

    def _predict(self, batch, viz_flag=False):
        """
        forward through the network
        """
        model_out = self.network(batch["model_input"], viz_flag)
        for k, v in model_out.items():
            batch[k] = v  # directly place all output to batch dict
        return batch

    def _postprocess(self, batch):
        """
        Additional operation process on one gpu or cpu
        :return: a dictionary
        """
        for k in self.output_specs["metric"]:
            try:
                batch[k] = batch[k].mean()
            except:
                # sometime the metric might not be computed, e.g. during training the val metric
                pass
        return batch

    def _postprocess_after_optim(self, batch):
        """
        Additional operation process after optimizer.step
        :return: a dictionary
        """
        return batch

    def _detach_before_return(self, batch):
        for k, v in batch.items():
            if isinstance(v, dict):
                self._detach_before_return(v)
            if isinstance(v, torch.Tensor):
                batch[k] = v.detach()
        return batch

    def train_batch(self, batch, viz_flag=False):
        batch = self._preprocess(batch, viz_flag)
        self.set_train()
        self.zero_grad()
        batch = self._predict(batch, viz_flag)
        batch = self._postprocess(batch)
        if self.loss_clip > 0.0:
            if abs(batch["batch_loss"]) > self.loss_clip:
                logging.warning(
                    f"Loss Clipped from {abs(batch['batch_loss'])} to {self.loss_clip}"
                )
            batch["batch_loss"] = torch.clamp(batch["batch_loss"], -self.loss_clip, self.loss_clip)
        batch["batch_loss"].backward()
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            if grad_norm > self.grad_clip:
                logging.info(
                    "Warning! Clip gradient from {} to {}".format(grad_norm, self.grad_clip)
                )
        self.optimizers_step()
        batch = self._postprocess_after_optim(batch)
        batch = self._detach_before_return(batch)
        return batch

    def val_batch(self, batch, viz_flag=False):
        batch = self._preprocess(batch, viz_flag)
        self.set_eval()
        with torch.no_grad():
            batch = self._predict(batch, viz_flag)
        batch = self._postprocess(batch)
        batch = self._dataparallel_postprocess(batch)
        batch = self._postprocess_after_optim(batch)
        batch = self._detach_before_return(batch)
        return batch

    def _dataparallel_postprocess(self, batch):
        if self.__dataparallel_flag__:
            for k in batch.keys():
                if k.endswith("loss") or k in self.output_specs["metric"]:
                    if isinstance(batch[k], list):
                        for idx in len(batch[k]):
                            batch[k][idx] = batch[k][idx].mean()
                    else:
                        batch[k] = batch[k].mean()
        return batch

    def zero_grad(self):
        for k in self.optimizer_dict.keys():
            self.optimizer_dict[k].zero_grad()

    def optimizers_step(self):
        for k in self.optimizer_dict.keys():
            self.optimizer_dict[k].step()

    def model_resume(self, checkpoint, is_initialization, network_name=None):
        # reprocess to fit the old version
        state_dict = {}
        logging.info("Load from ep {}".format(checkpoint["epoch"]))
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith("module."):
                name = ".".join(k.split(".")[1:])
            else:
                name = k
            state_dict[name] = v
        checkpoint["model_state_dict"] = state_dict
        if not is_initialization or network_name == ["all"]:
            self.network.load_state_dict(checkpoint["model_state_dict"], strict=True)
            for k, v in checkpoint["optimizers_state_dict"]:
                self.optimizer_dict[k].load_state_dict(v)
                # send to cuda
                for state in self.optimizer_dict[k].state.values():
                    for _k, _v in state.items():
                        if torch.is_tensor(_v):
                            state[_k] = _v.cuda()
        else:
            if network_name is not None:
                prefix = ["network_dict." + name for name in network_name]
                restricted_model_state_dict = {}
                for k, v in checkpoint["model_state_dict"].items():
                    for pf in prefix:
                        if k.startswith(pf):
                            restricted_model_state_dict[k] = v
                            break
                checkpoint["model_state_dict"] = restricted_model_state_dict
            self.network.load_state_dict(checkpoint["model_state_dict"], strict=False)

    def save_checkpoint(self, filepath, additional_dict=None):
        save_dict = {
            "model_state_dict": self.network.module.state_dict()
            if self.__dataparallel_flag__
            else self.network.state_dict(),
            "optimizers_state_dict": [
                (k, opti.state_dict()) for k, opti in self.optimizer_dict.items()
            ],
        }
        if additional_dict is not None:
            for k, v in additional_dict.items():
                save_dict[k] = v
        torch.save(save_dict, filepath)

    def to_gpus(self):
        if torch.cuda.device_count() > 1:
            self.network = nn.DataParallel(self.network)
            self.__dataparallel_flag__ = True
        else:
            self.__dataparallel_flag__ = False
        self.network.cuda()

    def set_train(self):
        self.network.train()

    def set_eval(self):
        self.network.eval()


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.network_dict = None

    def forward(self, *input):
        raise NotImplementedError
