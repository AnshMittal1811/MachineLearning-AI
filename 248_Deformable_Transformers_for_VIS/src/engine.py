"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
import os
import sys
import time
import json
from pathlib import Path
import tqdm
from zipfile import ZipFile
import numpy as np
import torch
from torch.utils.data import DataLoader
from src import trackeval
from src.datasets.vis import VISValDataset
import src.util.misc as utils
from src.datasets import get_coco_api_from_dataset
from src.datasets.coco_eval import CocoEvaluator
from src.datasets.panoptic_eval import PanopticEvaluator
from src.models import Tracker


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, visualizers: dict, vis_and_log_interval: int,
                    clip_max_norm: float):
    vis_iter_metrics = None
    if visualizers:
        vis_iter_metrics = visualizers['iter_metrics']

    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(
        vis_and_log_interval,
        delimiter="  ",
        vis=vis_iter_metrics,
        debug=False)
    metric_logger.add_meter('lr_base', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_backbone', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_linear_proj', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_mask_head', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_temporal_linear_proj', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, *_ = model(samples, targets)

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

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        metric_logger.update(lr_base=optimizer.param_groups[0]["lr"],
                             lr_backbone=optimizer.param_groups[1]["lr"],
                             lr_linear_proj=optimizer.param_groups[2]["lr"],
                             lr_mask_head=optimizer.param_groups[3]["lr"],
                             lr_temporal_linear_proj=optimizer.param_groups[4]["lr"]

                             )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(device=device)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_coco(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
                  data_loader: DataLoader, device, output_dir: Path, visualizers: dict,
                  vis_log_interval: int, epoch: int = None):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(
        vis_log_interval,
        delimiter="  ",
    )

    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    base_ds = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = tuple(k for k in ('bbox', 'segm') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 'Test:')):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs, *_ = model(samples, targets)

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

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for j, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[j]["image_id"] = image_id
                res_pano[j]["file_name"] = file_name

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
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in coco_evaluator.coco_eval:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in coco_evaluator.coco_eval:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    eval_stats = stats['coco_eval_bbox'][:3]
    if 'coco_eval_masks' in stats:
        eval_stats.extend(stats['coco_eval_masks'][:3])
    # VIS
    if visualizers:
        vis_epoch = visualizers['epoch_metrics']
        y_data = [stats[legend_name] for legend_name in vis_epoch.viz_opts['legend']]
        vis_epoch.plot(y_data, epoch)

        visualizers['epoch_eval'].plot(eval_stats, epoch)

    return eval_stats, coco_evaluator


@torch.no_grad()
def inference_vis(tracker: Tracker, data_loader_val: DataLoader, dataset_val: VISValDataset,
                  visualizers: dict, device: torch.device, output_dir: Path, out_folder_name: str,
                  epoch: int, selected_videos: str):

    tracker.model.eval()

    all_tracks = []
    init_time = time.time()
    all_times = []

    for idx, video in tqdm.tqdm(enumerate(data_loader_val)):
        if selected_videos and video.video_name not in selected_videos:
            continue
        video_tracks, all_times = tracker(video, device, all_times)
        all_tracks.extend(video_tracks)

    finish_time = time.time()
    print(f" Max memory allocated {int(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))}")
    print(f"Total time {finish_time - init_time}")

    # Compute FPS only if 1GPU is used, as otherwise some processes see padding videos
    if utils.get_world_size() == 1:
        print(f"Total time non-upsampling {sum(all_times)}")
        print(f"FPS: {dataset_val.get_total_num_frames() / sum(all_times)}")

    gathered_preds = utils.all_gather(all_tracks)
    results_accums = utils.accumulate_results(gathered_preds)

    class_av_ap_all, class_av_ar_all = None, None
    if dataset_val.has_gt:
        out_eval_folder = os.path.join(output_dir, "output_eval")
        if not os.path.exists(out_eval_folder):
            os.makedirs(out_eval_folder, exist_ok=True)
        class_av_ap_all, class_av_ar_all = evaluate_vis(results_accums, dataset_val.annotations, out_eval_folder)

    if out_folder_name or not dataset_val.has_gt:
        out_dir = os.path.join(output_dir, out_folder_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, "results.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results_accums, f)

        out_zip_filename = os.path.join(out_dir, "results.zip")
        with ZipFile(out_zip_filename, 'w') as zip_obj:
            zip_obj.write(out_file, os.path.basename(out_file))

    eval_stats = [class_av_ap_all, class_av_ar_all]

    # VIS
    if visualizers and class_av_ap_all is not None:
        visualizers['epoch_eval'].plot(eval_stats, epoch)

    return class_av_ap_all, class_av_ar_all


def evaluate_vis(results, gt_path, out_folder):
    # Eval config
    eval_config = trackeval.Evaluator.get_default_eval_config()
    # print only combined since TrackMAP is undefined for per sequence breakdowns
    eval_config['PRINT_ONLY_COMBINED'] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["OUTPUT_DETAILED"] = False
    eval_config["PLOT_CURVES"] = False
    eval_config["LOG_ON_ERROR"] = False

    # Dataset config
    dataset_config = trackeval.datasets.YouTubeVIS.get_default_dataset_config()
    dataset_config["PRINT_CONFIG"] = False

    dataset_config["OUTPUT_FOLDER"] = out_folder
    dataset_config["TRACKER_DISPLAY_NAMES"] = ["DeVIS"]
    dataset_config["TRACKERS_TO_EVAL"] = ["DeVIS"]

    # Metrics config
    metrics_config = {'METRICS': ['TrackMAP']}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.YouTubeVIS(dataset_config, gt=gt_path, predictions=results)]
    metrics_list = []
    for metric in [trackeval.metrics.TrackMAP, trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                   trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            # specify TrackMAP config for YouTubeVIS
            if metric == trackeval.metrics.TrackMAP:
                default_track_map_config = metric.get_default_metric_config()
                default_track_map_config['USE_TIME_RANGES'] = False
                default_track_map_config['AREA_RANGES'] = [[0 ** 2, 128 ** 2],
                                                           [128 ** 2, 256 ** 2],
                                                           [256 ** 2, 1e5 ** 2]]

                default_track_map_config['MAX_DETECTIONS'] = 100
                metrics_list.append(metric(default_track_map_config))
            else:
                metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    eval_results, eval_msg = evaluator.evaluate(dataset_list, metrics_list)
    clas_av = eval_results['YouTubeVIS']['DeVIS']['COMBINED_SEQ']['cls_comb_cls_av']['TrackMAP']
    # Mean score for each  Iou th from 0.5::0.95
    class_av_ap_all, class_av_ar_all = 100 * np.mean(clas_av["AP_all"]), 100 * np.mean(clas_av["AR_all"])

    return class_av_ap_all, class_av_ar_all
