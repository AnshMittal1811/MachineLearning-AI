from itertools import groupby
from numbers import Number
from typing import Union, Any, Optional, Dict, Tuple, List

import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import Tensor

from utils import scalars_to_log_dict, run_at_step, epoch_outputs_to_log_dict, is_rank_zero, get_rank


class BaseModule(LightningModule):
    def __init__(self):
        super().__init__()

    # Control flow
    def training_step(self, batch: Tuple[Tensor, dict], batch_idx: int, optimizer_idx: Optional[int] = None) -> Tensor:
        return self.shared_step(batch, batch_idx, optimizer_idx, 'train')

    def validation_step(self, batch: Tuple[Tensor, dict], batch_idx: int):
        return self.shared_step(batch, batch_idx, mode='validate')

    def test_step(self, batch: Tuple[Tensor, dict], batch_idx: int):
        return self.shared_step(batch, batch_idx, mode='test')

    def valtest_epoch_end(self, outputs: List[Dict[str, Tensor]], mode: str):
        if self.logger is None:
            return
        # Either log each step's output separately (results have been all_gathered in this case)
        if self.valtest_log_all:
            for image_dict in outputs:
                self._log_image_dict(image_dict, mode, commit=True)
        # Or just log a random batch worth of images from master process
        else:
            self._log_image_dict(epoch_outputs_to_log_dict(outputs, n_max="batch", shuffle=True), mode)

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]):
        self.valtest_epoch_end(outputs, 'validate')

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]):
        self.valtest_epoch_end(outputs, 'test')

    # Utility methods for logging
    def gather_tensor(self, t: Tensor) -> Tensor:
        return rearrange(self.all_gather(t), "m n c h w -> (m n) c h w")

    def gather_tensor_dict(self, d: Dict[Any, Tensor]) -> Dict[Any, Tensor]:
        return {k: rearrange(v.cpu(), "m n c h w -> (m n) c h w") for k, v in self.all_gather(d).items()}

    def log_scalars(self, scalars: Dict[Any, Union[Number, Tensor]], mode: str, **kwargs):
        if 'sync_dist' not in kwargs:
            kwargs['sync_dist'] = mode != 'train'
        self.log_dict(scalars_to_log_dict(scalars, mode), **kwargs)

    def _log_image_dict(self, img_dict: Dict[str, Tensor], mode: str, commit: bool = False, **kwargs):
        if self.logger is not None:
            for k, v in img_dict.items():
                self.logger.log_image_batch(f'{mode}/{k}', v, commit=commit, **kwargs)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)  # Improves performance

    def alert_nan_loss(self, loss: Tensor, batch_idx: int):
        if loss != loss:
            print(
                f'NaN loss in epoch {self.current_epoch}, batch index {batch_idx}, global step {self.global_step}, '
                f'local rank {get_rank()}. Skipping.')
        return loss != loss

    def _log_profiler(self):
        if run_at_step(self.trainer.global_step, self.log_timing_every_n_steps):
            report, total_duration = self.trainer.profiler._make_report()
            report_log = dict([kv for action, durations, duration_per in report for kv in
                               [(f'profiler/mean_t/{action}', np.mean(durations)),
                                (f'profiler/n_calls/{action}', len(durations)),
                                (f'profiler/total_t/{action}', np.sum(durations)),
                                (f'profiler/pct/{action}', duration_per)]])
            self.log_dict(report_log)
            self.logger.save_to_file('profiler_summary.txt', self.trainer.profiler.summary(), unique_filename=False)

    def on_train_start(self):
        if self.logger:
            self.logger.log_model_summary(self)

    def log_grad_norm(self, grad_norm_dict: Dict[str, torch.Tensor]) -> None:
        self.log_dict({'grads/' + k: v for k, v in grad_norm_dict.items()})

    def on_after_backward(self) -> None:
        if not getattr(self, 'validate_gradients', False):
            return

        valid_gradients = True
        invalid_params = []
        for name, param in self.named_parameters():
            if param.grad is not None:
                this_param_valid = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                valid_gradients &= this_param_valid
                if not this_param_valid:
                    invalid_params.append(name)
                # if not valid_gradients:
                #     break

        if not valid_gradients:
            depth_two_params = [k for k, _ in groupby(
                ['.'.join(n.split('.')[:2]).replace('.weight', '').replace('.bias', '') for n in invalid_params])]
            if is_rank_zero():
                print(f'Detected inf/NaN gradients for parameters {", ".join(depth_two_params)}. '
                      f'Skipping epoch {self.current_epoch}, batch index {self.batch_idx}, global step {self.global_step}.')
            self.zero_grad(set_to_none=True)
