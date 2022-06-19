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
from typing import Iterable

from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched

def compute_prf1(true_mask, pred_mask):
	"""
	Compute precision, recall, and F1 metrics for predicted mask against ground truth
	"""
	conf_mat = confusion_matrix(true_mask, pred_mask, labels=[False, True])
	p = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1] + 1e-8)
	r = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1] + 1e-8)
	f1 = (2*p*r) / (p+r+1e-8)
	return conf_mat, p, r, f1

# This always zips the two data loaders
def train_one_epoch_with_target(
	model: torch.nn.Module,
	criterion: torch.nn.Module,
	target_criterion: torch.nn.Module,
	source_data_loader: Iterable,
	target_data_loader: Iterable,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	loss_scaler,
	log_writer=None,
	args=None,
):
	model.train(True)
	metric_logger = misc.MetricLogger(delimiter="  ")
	metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
	header = "Epoch: [{}]".format(epoch)
	print_freq = 20

	accum_iter = args.accum_iter

	optimizer.zero_grad()
	if log_writer is not None:
		print("log_dir: {}".format(log_writer.log_dir))

	data_loader = zip(source_data_loader, target_data_loader)
	length = min(len(source_data_loader), len(target_data_loader))
	
	for data_iter_step, (
		((data_s, data_s_aug), label_s),
		((data_t, data_t_aug), label_t),
	) in enumerate(
		metric_logger.log_every(data_loader, print_freq, header, length=length)
	):

		# we use a per iteration (instead of per epoch) lr scheduler
		if data_iter_step % accum_iter == 0:
			lr_sched.adjust_learning_rate(
				optimizer, data_iter_step / length + epoch, args
			)

		data_s = data_s.to(device, non_blocking=True)
		label_s = label_s.to(device, non_blocking=True)

		data_t = data_t.to(device, non_blocking=True)

		data_s_aug = data_s_aug.to(device, non_blocking=True)
		data_t_aug = data_t_aug.to(device, non_blocking=True)

		label_t = label_t.to(device, non_blocking=True)

		with torch.cuda.amp.autocast():
			data_s_t = torch.cat([data_s_aug, data_t], dim=0)
			logits_s_t = model(data_s_t)
			logits_s = logits_s_t[:len(data_s)]
			classifier_loss_s = criterion(logits_s, label_s)
			loss = torch.zeros_like(classifier_loss_s)
			loss += classifier_loss_s

		# select target samples to use for self-training
		with torch.cuda.amp.autocast():
			logits_t = logits_s_t[len(data_s):]
			scores_t = F.softmax(logits_t, dim=1)
			top1_t, preds_t = scores_t.max(dim=1)

			if args.attention_seeding:
				attention_stride = args.committee_size
				attentions = model.module.get_last_selfattention(data_t_aug)
				nh = attentions.shape[1] # number of heads
				attentions = attentions[:, :, 0, 1:].reshape(len(data_t_aug), nh, -1)
				attention = attentions.mean(dim=1) # Average across heads
			else:
				attention_stride = 1
				attention = None

			sel_mask = torch.zeros_like(preds_t.detach()).long()
			exclude_mask = torch.zeros_like(attention).long()
			logits_t_correct = torch.zeros_like(logits_t)

			for _ in range(args.committee_size):
				mask_t, enc_feats_masked_t = model.module.forward_encoder(data_t_aug, args.mask_ratio, \
																				exclude_mask=exclude_mask, \
																				attention=attention, \
																				attention_stride=attention_stride)
				logits_masked_t = model.module.head(enc_feats_masked_t)
				preds_masked_t = logits_masked_t.argmax(dim=1, keepdim=True).squeeze()

				cur_sel_mask = (preds_masked_t == preds_t).detach()
				sel_mask += cur_sel_mask.type(torch.uint8)
				logits_t_correct[cur_sel_mask, :] = logits_masked_t[cur_sel_mask, :]
				exclude_mask = exclude_mask + (1-mask_t.long())
				if args.attention_seeding: attention_stride -= 1

			votes_req = args.committee_size # unanimous voting
			sel_mask_cons = (sel_mask >= votes_req)

			threshold = args.conf_threshold
			sel_mask_conf = (top1_t > threshold).detach()  # confidence thresholding
			sel_mask = torch.logical_or(sel_mask_cons, sel_mask_conf)

			correct_mask = (preds_t.detach() == label_t).cpu()
			_, correct_precision, _, _ = compute_prf1(correct_mask.cpu().numpy(), sel_mask.cpu().numpy())

			if sel_mask.sum() > 0:
				sel_ratio = sel_mask.sum().item()/len(sel_mask)
				sst_loss = args.lambda_unsup * sel_ratio * target_criterion(
					logits_t_correct[sel_mask], preds_t[sel_mask]
				)
				loss += sst_loss

		loss_value = loss.item()
		if not math.isfinite(loss_value):
			print("Loss is {}, stopping training".format(loss_value))
			sys.exit(1)

		loss_avg = loss / accum_iter

		loss_scaler(
			loss_avg,
			optimizer,
			parameters=model.parameters(),
			update_grad=(data_iter_step + 1) % accum_iter == 0,
		)

		if (data_iter_step + 1) % accum_iter == 0:
			optimizer.zero_grad()

		torch.cuda.synchronize()

		metric_logger.update(total_loss=loss_value)
		metric_logger.update(classifier_loss=classifier_loss_s.item())
		if sel_mask.sum() > 0:
			metric_logger.update(sst_loss=sst_loss.item())
			metric_logger.update(sel_ratio=sel_ratio)
			metric_logger.update(correct_precision=correct_precision)

		min_lr = 10.0
		max_lr = 0.0
		for group in optimizer.param_groups:
			min_lr = min(min_lr, group["lr"])
			max_lr = max(max_lr, group["lr"])

		metric_logger.update(lr=max_lr)

		loss_value_reduce = misc.all_reduce_mean(loss_value)
		classifier_loss_reduce = misc.all_reduce_mean(classifier_loss_s.item())
		if sel_mask.sum() > 0:
			sst_loss_reduce = misc.all_reduce_mean(sst_loss.item())
			sel_ratio_reduce = misc.all_reduce_mean(sel_ratio)
			correct_precision_reduce = misc.all_reduce_mean(correct_precision)

		if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
			"""We use epoch_1000x as the x-axis in tensorboard.
			This calibrates different curves when batch size changes.
			"""
			epoch_1000x = int((data_iter_step / length + epoch) * 1000)
			log_writer.add_scalar("train_total_loss", loss_value_reduce, epoch_1000x)
			log_writer.add_scalar(
				"train_classifier_loss", classifier_loss_reduce, epoch_1000x
			)
			if sel_mask.sum() > 0:
				log_writer.add_scalar("target_sst_loss", sst_loss_reduce, epoch_1000x)
				log_writer.add_scalar("sel_ratio", sel_ratio_reduce, epoch_1000x)
				log_writer.add_scalar("correct_precision", correct_precision_reduce, epoch_1000x)

			log_writer.add_scalar("lr", max_lr, epoch_1000x)

	# write_image(args, model, log_writer, recon_s_t, mask_s_t, data_masked_s_t, epoch * 1000)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


