# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from util.logging import log_stats_to_tboard, update_log_stats_dict

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
					data_loader: Iterable, optimizer: torch.optim.Optimizer,
					device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
					log_writer=None, args=None):
	model.train(True)
	metric_logger = misc.MetricLogger(delimiter="  ")
	metric_logger.add_meter('lr', misc.SmoothedValue(
		window_size=1, fmt='{value:.6f}'))
	header = 'Epoch: [{}]'.format(epoch)
	print_freq = 20

	accum_iter = args.accum_iter

	optimizer.zero_grad()

	if log_writer is not None:
		print('log_dir: {}'.format(log_writer.log_dir))

	for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

		# we use a per iteration (instead of per epoch) lr scheduler
		if data_iter_step % accum_iter == 0:
			lr_sched.adjust_learning_rate(
				optimizer, data_iter_step / len(data_loader) + epoch, args)

		samples = samples.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		with torch.cuda.amp.autocast():
			outputs = model(samples)
			loss = criterion(outputs, targets)

		loss_value = loss.item()

		if not math.isfinite(loss_value):
			print("Loss is {}, stopping training".format(loss_value))
			sys.exit(1)

		loss /= accum_iter
		loss_scaler(loss, optimizer, clip_grad=max_norm,
					parameters=model.parameters(), create_graph=False,
					update_grad=(data_iter_step + 1) % accum_iter == 0)
		if (data_iter_step + 1) % accum_iter == 0:
			optimizer.zero_grad()

		torch.cuda.synchronize()

		metric_logger.update(loss=loss_value)
		min_lr = 10.
		max_lr = 0.
		for group in optimizer.param_groups:
			min_lr = min(min_lr, group["lr"])
			max_lr = max(max_lr, group["lr"])

		metric_logger.update(lr=max_lr)

		loss_value_reduce = misc.all_reduce_mean(loss_value)
		if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
			""" We use epoch_1000x as the x-axis in tensorboard.
			This calibrates different curves when batch size changes.
			"""
			epoch_1000x = int(
				(data_iter_step / len(data_loader) + epoch) * 1000)
			log_writer.add_scalar('train_classifier_loss', loss_value_reduce, epoch_1000x)
			log_writer.add_scalar('lr', max_lr, epoch_1000x)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, num_classes=40):
	criterion = torch.nn.CrossEntropyLoss()

	metric_logger = misc.MetricLogger(delimiter="  ")

	# switch to evaluation mode
	model.eval()
	confusion_matrix = torch.zeros(num_classes, num_classes).long()

	# for batch in metric_logger.log_every(data_loader, 10000, header):
	for batch in data_loader:
		images = batch[0]
		target = batch[1]
		images = images.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)

		# compute output
		with torch.cuda.amp.autocast():
			output = model(images)
			if type(output) == tuple:
				output = output[-1]
			loss = criterion(output, target)
		
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		pred = output.argmax(dim=1, keepdim=True)
		
		for t, p in zip(target.view(-1), pred.view(-1)):
			confusion_matrix[t.long(), p.long()] += 1

		batch_size = images.shape[0]
		metric_logger.update(loss=loss.item())
		metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
		metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
		
	avg_acc = confusion_matrix.diagonal().numpy() / confusion_matrix.sum(axis=1).numpy()
	avg_acc = avg_acc.mean() * 100
	metric_logger.meters['avg_acc'].update(avg_acc, n=batch_size)
	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate_and_log(model, loader, device, num_imgs, epoch, log_writer, log_stats, name, args):
    test_stats = evaluate(loader, model, device, num_classes=args.nb_classes)
    print(f"Accuracy of the network on the {num_imgs} {name} images: {test_stats['acc1']:.1f}%")
    log_stats_to_tboard(log_writer, test_stats, epoch, name)
    update_log_stats_dict(log_stats, test_stats, epoch, name)
    return test_stats