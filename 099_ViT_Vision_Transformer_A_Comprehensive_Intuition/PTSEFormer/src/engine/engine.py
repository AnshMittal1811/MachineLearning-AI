# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, tb_writer=None, log_every_dir=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ", log_dir=log_every_dir)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    iteration = epoch * len(data_loader)
    # if isinstance(data_loader[0], dict):
    #
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        # for ret in data_loader:
        # print(type(targets))
        # print(samples.items())
        # for k,v in samples.items():
        #     print(k)
        # print(targets.keys())
        # samples = {k: [v_i.to(device, non_blocking=True) for v_i in v] for k, v in samples.items()}
        # samples = {'cur': samples['cur'].to(device, non_blocking=True), 'ref_l': [v_i.to(device, non_blocking=True) for v_i in samples['ref_l']]}
        # # print(samples['cur'].shape)
        # targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        # print(targets)
        # targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

    # for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        # print(ret)
        # if isinstance(ret, dict):
        #     print(ret['image'])
        #     samples = utils.NestedTensor(ret['image'], ret['pad_mask'])
        #     samples = samples.to(device)
        #     targets = {k: v.to(device) for k, v in ret.items() if
        #                k != 'orig_image' and k != 'image' and 'pad_mask' not in k}
        # else:
        #     samples, targets = ret
        #     samples = samples.to(device)
        #     targets = {k: v.to(device) for k, v in ret.items() if
        #                k != 'orig_image' and k != 'image' and 'pad_mask' not in k}
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()


        if tb_writer is not None:
            tb_writer.add_scalar(tag='train/' + 'loss', scalar_value=loss_value, global_step=iteration)
            for k, v in loss_dict_reduced_unscaled.items():
                tb_writer.add_scalar(tag='train/' + k, scalar_value=v, global_step=iteration)
            for k, v in loss_dict_reduced_scaled.items():
                tb_writer.add_scalar(tag='train/' + k, scalar_value=v, global_step=iteration)
            tb_writer.add_scalar(tag='train/' + 'grad_norm', scalar_value=grad_total_norm, global_step=iteration)
            tb_writer.add_scalar(tag='train/' + 'class_error', scalar_value=loss_dict_reduced['class_error'], global_step=iteration)


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()

        iteration += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, log_every_dir=None, ret_res=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ", log_dir=log_every_dir)
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    # if ret_res:
    res_dict = {}
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        tmp = {}
        for k, v in samples.items():
            if isinstance(v, list):
                tmp[k] = []
                for n_t in v:
                    tmp[k].append(n_t.to(device, non_blocking=True))
            else:
                tmp[k] = v.to(device, non_blocking=True)
        samples = tmp

        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # print(res)
        if ret_res:
            res_dict.update(res)
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        stats_dict = coco_evaluator.summarize()
        # print(stats_dict)
        if utils.is_main_process():
            with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
                # print(stats_dict.values())
                f.write(str(stats_dict.values()) + '\n')
                # np.savetxt(f, stats_dict.values(), newline='\n')

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    if ret_res:
        return stats, coco_evaluator, res_dict
    return stats, coco_evaluator


# @torch.no_grad()
# def evaluate_exp(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, log_every_dir=None, ret_res=False):
#     model.eval()
#     criterion.eval()
#
#     metric_logger = utils.MetricLogger(delimiter="  ", log_dir=log_every_dir)
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Test:'
#
#     iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
#     coco_evaluator = CocoEvaluator(base_ds, iou_types)
#     # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
#
#     panoptic_evaluator = None
#     if 'panoptic' in postprocessors.keys():
#         panoptic_evaluator = PanopticEvaluator(
#             data_loader.dataset.ann_file,
#             data_loader.dataset.ann_folder,
#             output_dir=os.path.join(output_dir, "panoptic_eval"),
#         )
#     # if ret_res:
#     res_dict = {}
#     for samples, targets in metric_logger.log_every(data_loader, 10, header):
#         # samples = samples.to(device)
#         # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         tmp = {}
#         for k, v in samples.items():
#             if isinstance(v, list):
#                 tmp[k] = []
#                 for n_t in v:
#                     tmp[k].append(n_t.to(device, non_blocking=True))
#             else:
#                 tmp[k] = v.to(device, non_blocking=True)
#         samples = tmp
#
#         targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
#
#         outputs = model(samples)
#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict
#
#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])
#
#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes)
#         if 'segm' in postprocessors.keys():
#             target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#             results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         print(res)
#         if ret_res:
#             res_dict.update(res)
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)
#
#         if panoptic_evaluator is not None:
#             res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#             for i, target in enumerate(targets):
#                 image_id = target["image_id"].item()
#                 file_name = f"{image_id:012d}.png"
#                 res_pano[i]["image_id"] = image_id
#                 res_pano[i]["file_name"] = file_name
#
#             panoptic_evaluator.update(res_pano)
#
#         boxes = res["boxes"]
#         scores = res["scores"]
#         labels = res["labels"]
#
#         # print(scores)
#         # exit()
#
#         def filter_low_conf(scores):
#             keep = np.where(np.array(scores) > 0.3)
#             print(keep)
#             return keep[0]
#
#         import cv2
#         img_curr_nest = samples['cur']
#         input_tensor, _ = img_curr_nest.decompose()
#         _input_tensor = input_tensor.detach()[0].cpu().permute(1, 2, 0).numpy()
#         # mem_save = np.sum(np.abs(mem_save), axis=0)
#         _input_tensor = (_input_tensor - np.min(_input_tensor)) / (np.max(_input_tensor) - np.min(_input_tensor)) * 255
#         image = cv2.cvtColor(_input_tensor, cv2.COLOR_RGB2BGR)
#         # cv2.imwrite(filename, _input_tensor)
#
#         keep = filter_low_conf(scores)
#
#         boxes = [boxes[i] for i in keep]
#         scores = [scores[i] for i in keep]
#         labels = [labels[i] for i in keep]
#
#         result = image.copy()
#         result = overlay_boxes(result, boxes, labels)
#         result = overlay_class_names(result, boxes, labels, scores)
#         if save_dir:
#             save_img(result, idx)
#         return result
#
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#     if panoptic_evaluator is not None:
#         panoptic_evaluator.synchronize_between_processes()
#
#     # accumulate predictions from all images
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         stats_dict = coco_evaluator.summarize()
#         # print(stats_dict)
#         if utils.is_main_process():
#             with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
#                 # print(stats_dict.values())
#                 f.write(str(stats_dict.values()) + '\n')
#                 # np.savetxt(f, stats_dict.values(), newline='\n')
#
#     panoptic_res = None
#     if panoptic_evaluator is not None:
#         panoptic_res = panoptic_evaluator.summarize()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     if coco_evaluator is not None:
#         if 'bbox' in postprocessors.keys():
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in postprocessors.keys():
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#     if panoptic_res is not None:
#         stats['PQ_all'] = panoptic_res["All"]
#         stats['PQ_th'] = panoptic_res["Things"]
#         stats['PQ_st'] = panoptic_res["Stuff"]
#
#
#
#     if ret_res:
#         return stats, coco_evaluator, res_dict
#     return stats, coco_evaluator
