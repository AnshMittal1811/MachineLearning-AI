import argparse
import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple, Dict

from lib import modeling, metrics, visualize

from lib.data import setup_dataloader

from lib.modeling.utils import thicken_grid
from lib.visualize.mesh import get_mesh

from lib.config import config
from lib.structures.field_list import collect


from tools.test_net_single_image import configure_inference
from lib.utils import re_seed


def main(opts, start=0, end=None):
    configure_inference(opts)

    re_seed(0)
    device = torch.device("cuda:0")

    output_path = Path(config.OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)

    # Define model and load checkpoint.
    print("Load model...")
    model = modeling.PanopticReconstruction()
    checkpoint = torch.load(opts.model)
    model.load_state_dict(checkpoint["model"])  # load model checkpoint
    model = model.to(device)  # move to gpu
    model.switch_test()

    # Define dataset
    # config.DATALOADER.NUM_WORKERS=0
    dataloader = setup_dataloader(config.DATASETS.VAL, False, is_iteration_based=False, shuffle=False)
    dataloader.dataset.samples = dataloader.dataset.samples[start:end]
    print(f"Loaded {len(dataloader.dataset)} samples.")

    # Prepare metric
    metric = metrics.PanopticReconstructionQuality()

    for idx, (image_ids, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if targets is None:
            print(f"Error, {image_ids[0]}")
            continue

        # Get input images
        images = collect(targets, "color")

        # Pass through model
        with torch.no_grad():
            try:
                _, results = model(images, targets)
            except Exception as e:
                print(e)
                del targets, images
                continue

        if config.DATASETS.NAME == "front3d":
            frustum_mask = dataloader.dataset.frustum_mask
        else:
            frustum_mask = targets[0].get_field("frustum_mask").squeeze()

        # Prepare ground truth
        instances_gt, instance_semantic_classes_gt = _prepare_semantic_mapping(targets[0].get_field("instance3d").squeeze(),
                                                                               targets[0].get_field("semantic3d").squeeze())
        distance_field_gt = targets[0].get_field("geometry").squeeze()
        instance_information_gt = _prepare_instance_masks_thicken(instances_gt, instance_semantic_classes_gt,
                                                                  distance_field_gt, frustum_mask)

        # Prepare prediction
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        distance_field_pred = results["frustum"]["geometry"].dense(dense_dimensions, min_coordinates, default_value=truncation)[0].to("cpu", non_blocking=False)
        instances_pred = results["panoptic"]["panoptic_instances"].to("cpu", non_blocking=False)

        instance_semantic_classes_pred = results["panoptic"]["panoptic_semantic_mapping"]
        instance_information_pred = _prepare_instance_masks_thicken(instances_pred, instance_semantic_classes_pred,
                                                                    distance_field_pred, frustum_mask)

        # Add to metric
        # Format: Dict[instance_id: instance_mask, semantic_label]
        per_sample_result = metric.add(instance_information_pred, instance_information_gt)

        file_name = image_ids[0].replace("/", "_")
        with open(output_path / f"{file_name}.json", "w") as f:
            json.dump({k: cat.as_metric for k, cat in per_sample_result.items()}, f, indent=4)

        if opts.verbose:
            pprint(per_sample_result)

    # Reduce metric
    quantitative = metric.reduce()

    # Print results
    for k, v in quantitative.items():
        print(f"{k:>5}", f"{v:.3f}")


def _prepare_instance_masks_thicken(instances, semantic_mapping, distance_field, frustum_mask) -> Dict[int, Tuple[torch.Tensor, int]]:
    instance_information = {}

    for instance_id, semantic_class in semantic_mapping.items():
        instance_mask: torch.Tensor = (instances == instance_id)
        instance_distance_field = torch.full_like(instance_mask, dtype=torch.float, fill_value=3.0)
        instance_distance_field[instance_mask] = distance_field.squeeze()[instance_mask]
        instance_distance_field_masked = instance_distance_field.abs() < 1.0

        # instance_grid = instance_grid & frustum_mask
        instance_grid = thicken_grid(instance_distance_field_masked, [256, 256, 256], frustum_mask)
        instance_information[instance_id] = instance_grid, semantic_class

    return instance_information


def _prepare_semantic_mapping(instances, semantics, offset=2):
    semantic_mapping = {}
    panoptic_instances = torch.zeros_like(instances).int()

    things_start_index = offset  # map wall and floor to id 1 and 2

    unique_instances = instances.unique()
    for index, instance_id in enumerate(unique_instances):
        # Ignore freespace
        if instance_id != 0:
            # Compute 3d instance surface mask
            instance_mask: torch.Tensor = (instances == instance_id)
            # instance_surface_mask = instance_mask & surface_mask
            panoptic_instance_id = index + things_start_index
            panoptic_instances[instance_mask] = panoptic_instance_id

            # get semantic prediction
            semantic_region = torch.masked_select(semantics, instance_mask)
            semantic_things = semantic_region[
                (semantic_region != 0) & (semantic_region != 10) & (semantic_region != 11)]

            unique_labels, semantic_counts = torch.unique(semantic_things, return_counts=True)
            max_count, max_count_index = torch.max(semantic_counts, dim=0)
            selected_label = unique_labels[max_count_index]

            semantic_mapping[panoptic_instance_id] = selected_label.int().item()

    # Merge stuff classes
    # Merge floor class
    wall_class = 10
    wall_id = 1
    wall_mask = semantics == wall_class
    panoptic_instances[wall_mask] = wall_id
    semantic_mapping[wall_id] = wall_class

    # Merge floor class
    floor_class = 11
    floor_id = 2
    floor_mask = semantics == floor_class
    panoptic_instances[floor_mask] = floor_id
    semantic_mapping[floor_id] = floor_class

    return panoptic_instances, semantic_mapping


def evaluate_jsons(opts):
    result_path = Path(opts.output)
    samples = [file for file in result_path.iterdir() if file.suffix == ".json"]

    print(f"Found {len(samples)} samples")
    metric = metrics.PanopticReconstructionQuality()
    for sample in tqdm(samples):
        try:
            content = json.load(open(sample))
            data = {}
            for k, cat in content.items():
                panoptic_sample = metrics.PQStatCategory()
                panoptic_sample.iou = cat["iou"]
                panoptic_sample.tp = cat["tp"]
                panoptic_sample.fp = cat["fp"]
                panoptic_sample.fn = cat["fn"]
                panoptic_sample.n = cat["n"]
                data[int(k)] = panoptic_sample

            metric.add_sample(data)
        except Exception as e:
            print(f"Error with {sample}")
            continue

    summary = metric.reduce()

    for name, value in summary.items():
        if name[0] == "n":
            print(f"{name:>10}\t\t{value:>5d}")
        else:
            print(f"{name:>10}\t\t{value:>5.3f}")

    with open(result_path / "panoptic_result.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="output/evaluation/")
    parser.add_argument("--config-file", "-c", type=str, default="configs/front3d_evaluate.yaml")
    parser.add_argument("--model", "-m", type=str, default="resources/panoptic-front3d.pth")
    parser.add_argument("-s", type=int, default=0)
    parser.add_argument("-e", type=int, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.eval_only:
        evaluate_jsons(args)
    else:
        main(args, args.s, args.e)
