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
from itertools import chain
from collections import Counter
import torch
from tqdm import tqdm
import random
import torch.nn as nn
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP


def normalize_image_per_patch(model, img):
    patches = model.patchify(img)
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True)
    norm_patches = (patches - mean) / (var + 1.e-6)**.5
    norm_img = model.unpatchify(norm_patches)
    return norm_img


def unnormalize_image_per_patch(model, img, gt):
    patches = model.patchify(gt)
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True)
    unnorm_patches = (img * (var + 1e-6)**.5) + mean
    unnorm_img = model.unpatchify(unnorm_patches)
    return unnorm_img


def contrast_stretch(img):
    # first dimension has to be batch dim
    img_view = img.view(img.shape[0], -1)
    img_max = img_view.max(1, keepdim=True)[0]
    img_min = img_view.min(1, keepdim=True)[0]
    img_view = (img_view - img_min) / (img_max - img_min)
    return img_view.view(img.shape)


def write_image(args, model, log_writer, pred, mask, samples, step):
    if log_writer is not None:
        if isinstance(model, DDP):
            model = model.module

        norm_gt = normalize_image_per_patch(model, samples)
        unnorm_pred_img = unnormalize_image_per_patch(model, pred, samples)
        pred_img = model.unpatchify(pred)

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        masked = samples * (1 - mask)
        im_paste = samples * (1 - mask) + pred_img * mask

        images = []
        # select 5 images to be logged
        n_samples = min(5, samples.shape[0])
        indices = random.sample(range(samples.shape[0]), n_samples)

        im_paste = contrast_stretch(im_paste)
        pred_img = contrast_stretch(pred_img)
        samples = contrast_stretch(samples)
        norm_gt = contrast_stretch(norm_gt)
        unnorm_pred_img = contrast_stretch(unnorm_pred_img)
        masked = contrast_stretch(masked)

        for i in indices:
            images += [masked[i], pred_img[i], im_paste[i], samples[i]]
            if args.norm_pix_loss:
                images += [norm_gt[i], unnorm_pred_img[i], samples[i]]

        grid = torchvision.utils.make_grid(images, nrow=len(images)//n_samples)
        log_writer.add_image('images', grid, step)


def train_one_epoch(model: torch.nn.Module,
                    source_data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, target_data_loader : Iterable = None,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if target_data_loader is None:
        data_loader = source_data_loader
    else:
        data_loader = list(zip(source_data_loader, target_data_loader))
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header, length=len(data_loader))):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / (len(data_loader)) + epoch, args)

        if target_data_loader is None:
            samples, _ = data
            samples = samples.to(device, non_blocking=True)

            # with torch.cuda.amp.autocast():
            loss, pred, mask = model(
                samples, mask_ratio=args.mask_ratio, decoder_ix=0)

        else:
            samples_s = data[0][0]
            samples_t = data[1][0]
            samples_s = samples_s.to(device, non_blocking=True)
            samples_t = samples_t.to(device, non_blocking=True)

            loss_s, pred_s, mask_s = model(
            samples_s, mask_ratio=args.mask_ratio)
            loss_t, pred_t, mask_t = model(
            samples_t, mask_ratio=args.mask_ratio)

            samples = torch.cat([samples_s, samples_t])
            pred = torch.cat([pred_s, pred_t])
            mask = torch.cat([mask_s, mask_t])
            loss = loss_s + loss_t

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Log images
        if data_iter_step % args.img_log_freq == 0:
            epoch_1000x = int(
                (data_iter_step / (len(data_loader)) + epoch) * 1000)
            write_image(args, model, log_writer, pred, mask, samples, epoch_1000x)

        loss_avg = loss / accum_iter
        loss_avg.backward(create_graph=False)
        # loss_scaler(loss_avg, optimizer, parameters=model.parameters(), clip_grad=None,
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / (len(data_loader)) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    write_image(args, model, log_writer, pred, mask, samples, epoch * 1000)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate_knn_accuracy(model, source_loader, target_loader, device, args):
    # TODO: C=65 only for OfficeHome
    C = 65
    src_embedding = torch.zeros([len(source_loader.dataset), 768]).to(device)
    src_labels = torch.zeros(len(source_loader.dataset)).long().to(device)
    model.eval()
    model_select = model.module if args.distributed else model
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(source_loader):
            data = data.to(device)
            latent_s, _, ids_restore_s = model_select.forward_encoder(data, mask_ratio=0.0)            
            emb = latent_s[:, 0, :]
            
            src_embedding[batch_idx*args.batch_size:batch_idx*args.batch_size+data.size(0), :] = emb
            src_labels[batch_idx*args.batch_size:batch_idx*args.batch_size+data.size(0)] = target.to(device)

    K = 7
    total, top1 = 0, 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(target_loader):            
            inputs = inputs.to(device)
            batchSize = inputs.size(0)
            latent_t, _, ids_restore_t = model_select.forward_encoder(inputs, mask_ratio=0.0)
            emb = latent_t[:, 0, :]
            dist = -torch.cdist(emb, src_embedding)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            pred_labels, _ = torch.mode(src_labels[yi], dim=-1)
            correct = pred_labels.eq(targets.to(device))
            top1 = top1 + correct.sum().item()
            total += targets.size(0)
    top1_acc = top1*100. / total

    print('tgt CD-kNN acc. on {} tgt examples: Top-1={:.2f}%'.format(total, top1_acc))
    return top1_acc
