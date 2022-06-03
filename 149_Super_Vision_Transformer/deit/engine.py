# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from deit.losses import DistillationLoss
import utils
import random
import torch.nn.functional as F

import pdb

def get_training_sub_network(img_size_list = [128, 160, 192, 224], keep_ratio_list=[0.5, 0.7, 1]):
    sub_network = []
    sub_network.append((224,1))
    keep_ratio = random.choice(keep_ratio_list[:-1])
    sub_network.append((224,keep_ratio))
    size = random.choice(img_size_list[:-1])
    sub_network.append((size,1))
    keep_ratio = random.choice(keep_ratio_list[:-1])
    sub_network.append((size,keep_ratio))
    return sub_network


def get_testing_sub_network(img_size_list = [128, 160, 192, 224], keep_ratio_list=[0.5, 0.7, 1]):
    sub_network = []
    for img_size in img_size_list:
        for keep_ratio in keep_ratio_list:
            sub_network.append((img_size, keep_ratio))
    return sub_network




def super_train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    kl_layer = torch.nn.KLDivLoss(reduction='batchmean')

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        sub_networks = get_training_sub_network()
        # print(sub_networks)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        optimizer.zero_grad()

        for index,sub_network in enumerate(sub_networks):
            with torch.cuda.amp.autocast():
                outputs = model(samples, sub_network[0], sub_network[1])
                if index%2 == 0:
                    # print(f'CE:{sub_network}')
                    loss = criterion(samples, outputs, targets)
                    max_outputs = outputs
                    max_output_detach_softmax = F.softmax(max_outputs.detach(), dim=1)
                else:
                    # print(f'KL:{sub_network}')
                    log_softmax_result = F.log_softmax(outputs, dim=1)
                    loss = kl_layer(log_softmax_result, max_output_detach_softmax)

            loss_value = loss.item()
            if index == 0:
                max_subnetwork_loss = loss_value

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)


            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order)

        loss_scaler._scaler.step(optimizer)
        loss_scaler._scaler.update()
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=max_subnetwork_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def super_evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    sub_networks = get_testing_sub_network()
    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        for sub_network in sub_networks:
            with torch.cuda.amp.autocast():
                output = model(images, sub_network[0], sub_network[1])

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.meters[f'{sub_network[0]}_{sub_network[1]}_acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters[f'{sub_network[0]}_{sub_network[1]}_acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for sub_network in sub_networks:
        print('* {image_size}_{keep_ratio}_Acc@1 {top1.global_avg:.3f} {image_size}_{keep_ratio}_Acc@5 {top5.global_avg:.3f} '
            .format(image_size=sub_network[0], keep_ratio=sub_network[1], top1=metric_logger.meters[f'{sub_network[0]}_{sub_network[1]}_acc1'], top5=metric_logger.meters[f'{sub_network[0]}_{sub_network[1]}_acc5']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

