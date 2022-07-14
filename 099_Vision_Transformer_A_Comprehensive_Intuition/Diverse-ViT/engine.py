# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""

import sys
import math
import utils
import torch
import torch.nn as nn
from timm.data import Mixup
from losses import DistillationLoss
from typing import Iterable, Optional
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from reg import *

def train_one_epoch_diverse(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, patch_targets, targets = mixup_fn(samples, targets)
            patch_targets = patch_targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs, h_first, h_last, attn_first = model(samples)
            loss = criterion(samples, outputs[:,0], targets)

            if not args.mixing_coef == 0:
                loss_mix, patch_num = Loss_mixing(outputs, patch_targets)
                loss = (loss + loss_mix) / patch_num

            if not args.emb_cos_within_coef == 0:
                loss_diverse = Loss_cosine(h_last)
                loss += args.emb_cos_within_coef * loss_diverse

            if not args.emb_contrast_cross_coef == 0:
                loss_diverse = Loss_contrastive(h_first, h_last)
                loss += args.emb_contrast_cross_coef * loss_diverse

            if not args.attn_cos_within_coef == 0:
                loss_diverse = Loss_cosine_attn(attn_first)
                loss += args.attn_cos_within_coef * loss_diverse

            if not args.weight_mha_cond_orth_coef == 0:
                loss_diverse = 0
                for pname, pweight in model.named_parameters():
                    if 'attn.qkv.weight' in pname:
                        dim = pweight.shape[-1]
                        new_weight = pweight.reshape(3, dim, dim)
                        qw, kw, vw = new_weight[0,:,:], new_weight[1,:,:], new_weight[2,:,:]
                        qloss = Loss_condition_orth_weight(qw)
                        loss_diverse += qloss
                        kloss = Loss_condition_orth_weight(kw)
                        loss_diverse += kloss
                        vloss = Loss_condition_orth_weight(vw)
                        loss_diverse += vloss
                loss += args.weight_mha_cond_orth_coef*loss_diverse

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("* Loss is {}, skip current iteration".format(loss_value))
            loss = torch.nan_to_num(loss)
            loss_value = loss.item()
            # sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

