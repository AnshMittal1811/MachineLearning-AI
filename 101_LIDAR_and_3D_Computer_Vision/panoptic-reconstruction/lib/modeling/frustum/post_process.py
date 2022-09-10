from datetime import datetime
from typing import List, Dict

import torch

import MinkowskiEngine as Me

from lib.config import config
from torch import nn

from lib.modeling.utils import ModuleResult


# TODO: assumes batch size = 0

class PostProcess(nn.Module):
    def __init__(self, things_classes: List[int] = None, stuff_classes: List[int] = None) -> None:
        super().__init__()

        self.thing_classes = things_classes
        if things_classes is None:
            self.thing_classes = []

        self.stuff_classes = stuff_classes
        if stuff_classes is None:
            self.stuff_classes = []

    def forward(self, instance_data: Dict[str, torch.Tensor], frustum_data: Dict[str, Me.SparseTensor]) -> ModuleResult:
        # dense
        device = frustum_data["instance3d"].device
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

        geometry, _, _ = frustum_data["geometry"].dense(dense_dimensions, min_coordinates, default_value=truncation)
        instances, _, _ = frustum_data["instance3d"].dense(dense_dimensions, min_coordinates)
        semantics, _, _ = frustum_data["semantic3d_label"].dense(dense_dimensions, min_coordinates)

        geometry = geometry.squeeze()
        instances = instances.squeeze()
        semantics = semantics.squeeze()

        # filter 3d instances by 2d instances
        instances_filtered = filter_instances(instance_data, instances)

        # merge output
        panoptic_instances = torch.zeros_like(geometry).int()
        panoptic_semantic_mapping = {}

        things_start_index = 2  # map wall and floor to id 1 and 2

        surface_mask = geometry.abs() <= 1.5

        # Merge things classes
        for index, instance_id in enumerate(instances_filtered.unique()):
            # Ignore freespace
            if instance_id != 0:
                # Compute 3d instance surface mask
                instance_mask: torch.Tensor = (instances == instance_id)

                if instance_mask.sum() > 0:
                    # instance_surface_mask = instance_mask & surface_mask
                    panoptic_instance_id = index + things_start_index
                    panoptic_instances[instance_mask] = panoptic_instance_id

                    # get semantic prediction
                    semantic_region = torch.masked_select(semantics, instance_mask)
                    semantic_things = semantic_region[(semantic_region != 0) & (semantic_region != 10) & (semantic_region != 11)]

                    unique_labels, semantic_counts = torch.unique(semantic_things, return_counts=True)
                    max_count, max_count_index = torch.max(semantic_counts, dim=0)
                    selected_label = unique_labels[max_count_index]

                    panoptic_semantic_mapping[panoptic_instance_id] = selected_label.int().item()


        # Merge stuff classes
        # Merge floor class
        wall_class = 10
        wall_id = 1
        wall_mask = semantics == wall_class
        panoptic_instances[wall_mask] = wall_id
        panoptic_semantic_mapping[wall_id] = wall_class

        # Merge floor class
        floor_class = 11
        floor_id = 2
        floor_mask = semantics == floor_class
        panoptic_instances[floor_mask] = floor_id
        panoptic_semantic_mapping[floor_id] = floor_class

        # Search label for unassigned surface voxels
        unassigned_voxels = (surface_mask & (panoptic_instances == 0).bool()).nonzero()

        panoptic_instances_copy = panoptic_instances.clone()
        # for voxel in unassigned_voxels:
        #     label = nn_search_old(panoptic_instances_copy, voxel)
        #
        #     panoptic_instances[voxel[0], voxel[1], voxel[2]] = label

        panoptic_instances[
            unassigned_voxels[:, 0],
            unassigned_voxels[:, 1],
            unassigned_voxels[:, 2]] = nn_search(panoptic_instances_copy, unassigned_voxels)

        panoptic_semantics = torch.zeros_like(panoptic_instances)

        for instance_id, semantic_label in panoptic_semantic_mapping.items():
            instance_mask = panoptic_instances == instance_id
            panoptic_semantics[instance_mask] = semantic_label

        result = {"panoptic_instances": panoptic_instances, "panoptic_semantics": panoptic_semantics,
                  "panoptic_semantic_mapping": panoptic_semantic_mapping}

        return {}, result


def filter_instances(instances2d,  instances3d):
    instances_filtered = torch.zeros_like(instances3d)
    instance_ids_2d = (instances2d["locations"][0] + 1)
    for instance_id in instance_ids_2d:
        if instance_id != 0:
            instance_mask = instances3d == instance_id
            instances_filtered[instance_mask] = instance_id

    return instances_filtered


def nn_search_old(grid, point, radius=3):
    start = -radius
    end = radius

    for x in range(start, end):
        for y in range(start, end):
            for z in range(start, end):
                offset = torch.tensor([x, y, z], device=point.device)
                point_offset = point + offset
                label = grid[point_offset[0], point_offset[1], point_offset[2]]

                if label != 0:
                    return label

    return 0

def nn_search(grid, point, radius=3):
    start = -radius
    end = radius
    label = torch.zeros([len(point)], device=point.device, dtype=grid.dtype)
    mask = torch.zeros_like(label).bool()

    for x in range(start, end):
        for y in range(start, end):
            for z in range(start, end):
                offset = torch.tensor([x, y, z], device=point.device)
                point_offset = point + offset
                label_bi = grid[point_offset[:, 0],
                                point_offset[:, 1],
                                point_offset[:, 2]]

                if label_bi.sum() != 0:
                    new_mask = (label_bi > 0) * (~mask)
                    label[new_mask] = label_bi[new_mask]
                    mask = mask + new_mask
    return label

