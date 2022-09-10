from typing import List, Tuple

import torch
from lib.structures.field_list import FieldList
from torch import nn
from torch.nn import functional as F

import MinkowskiEngine as Me

from lib.config import config
from lib.structures import DepthMap
from lib.structures.frustum import generate_frustum, generate_frustum_volume, compute_camera2frustum_transform


# TODO: clean up


class SparseProjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        self.image_size = (160, 120)
        self.depth_min = 0.4
        self.depth_max = 6.0
        self.voxel_size = config.MODEL.PROJECTION.VOXEL_SIZE
        self.frustum_dimensions = torch.tensor([256, 256, 256])

    def forward(self, depth, features, instances, targets) -> Me.SparseTensor:
        device = depth.device
        batch_size = depth.size(0)

        sparse_coordinates = []
        sparse_features = []

        # Process each sample in the batch individually
        for idx in range(batch_size):
            # Get GT intrinsic matrix
            intrinsic = targets[idx].get_field("depth").intrinsic_matrix
            camera2frustum = compute_camera2frustum_transform(intrinsic.cpu(), self.image_size, self.depth_min,
                                                              self.depth_max, self.voxel_size)
            camera2frustum = camera2frustum.to(device)

            intrinsic = intrinsic.to(device)
            intrinsic_inverse = torch.inverse(intrinsic)

            # Mask out depth pixels which fall into another room
            if config.MODEL.PROJECTION.FILTER_ROOM_PIXELS:
                room_mask = targets[idx].get_field("room_mask").to(device, non_blocking=True).squeeze()
                depth = depth.clone()
                depth[idx, 0, room_mask != True] = 0

            depth_pixels_xy = depth[idx, 0].nonzero(as_tuple=False)
            if depth_pixels_xy.shape[0] == 0:
                continue

            depth_pixels_z = depth[idx, 0][depth_pixels_xy[:, 0], depth_pixels_xy[:, 1]].reshape(-1)

            yv = depth_pixels_xy[:, 0].reshape(-1).float() * depth_pixels_z.float()
            xv = depth_pixels_xy[:, 1].reshape(-1).float() * depth_pixels_z.float()

            depth_pixels = torch.stack([xv, yv, depth_pixels_z.float(), torch.ones_like(depth_pixels_z).float()])
            pointcloud = torch.mm(intrinsic_inverse, depth_pixels.float())
            grid_coordinates = torch.mm(camera2frustum, pointcloud).t()[:, :3].contiguous()

            # projective sdf encoding
            # repeat truncation, add / subtract z-offset
            num_repetition = int(self.truncation * 2) + 1
            grid_coordinates = grid_coordinates.unsqueeze(1).repeat(1, num_repetition, 1)
            voxel_offsets = torch.arange(-self.truncation, self.truncation + 1, 1.0, device=device).view(1, -1, 1)
            coordinates_z = grid_coordinates[:, :, 2].clone()
            grid_coordinates[:, :, 2] += voxel_offsets[:, :, 0]

            num_points = grid_coordinates.size(0)

            if num_points == 0:
                continue

            df_values = coordinates_z - coordinates_z.int()
            df_values = df_values + voxel_offsets.squeeze(-1)
            df_values.unsqueeze_(-1)

            # encode sign and values in 2 different channels
            if config.MODEL.PROJECTION.SIGN_CHANNEL:
                sign = torch.sign(df_values)
                value = torch.abs(df_values)
                df_values = torch.cat([sign, value], dim=-1)

            sample_features = []

            # image features
            image_features = features[idx, :, depth_pixels_xy[:, 0], depth_pixels_xy[:, 1]]
            image_features = image_features.permute(1, 0)
            sample_features.append(image_features)

            # instance features
            mask_logits = instances["raw"]  # use mask logits
            locations = instances["locations"]
            start_channel = 1  # start at 1 to have freespace at location 0
            num_instance_features = config.MODEL.INSTANCE2D.MAX + start_channel
            mask = mask_logits[idx]
            location = locations[idx].long()

            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            instance_tensor = torch.zeros((1, num_instance_features, 120, 160))
            label_mask = torch.zeros(120, 160)
            sample_labels2d = [0]

            # place instance features at correct index in 3d grid
            for instance_index, position in enumerate(location):
                if len(mask) > 0 and instance_index < len(mask) and instance_index < num_instance_features and position < num_instance_features:
                    resized_mask = F.interpolate(input=mask[instance_index].unsqueeze(0).unsqueeze(0).float(), size=(120, 160), mode="nearest").squeeze(0).squeeze(0)
                    instance_tensor[0, position + start_channel] = resized_mask
                    label_mask += resized_mask
                    sample_labels2d.append(position + start_channel)

            instance_tensor[0, 0, label_mask == 0] = 1.0  # set freespace softmax label
            instance_tensor = instance_tensor.to(device)
            instance_features = instance_tensor[0, :, depth_pixels_xy[:, 0], depth_pixels_xy[:, 1]]
            instance_features = instance_features.permute(1, 0)
            sample_features.append(instance_features)

            if sample_features:
                sample_features = torch.cat(sample_features, dim=-1)
                sample_features = sample_features.unsqueeze(1).repeat(1, num_repetition, 1)
                sample_features = torch.cat([df_values, sample_features], dim=-1)
            else:
                sample_features = df_values

            # flatten repeated coordinates and features
            flatten_coordinates = grid_coordinates.view(num_points * num_repetition, 3)

            # pad to 256,256,256
            padding_offsets = self.compute_frustum_padding(intrinsic_inverse)
            flatten_coordinates = flatten_coordinates + padding_offsets #- torch.tensor([1, 1, 1]).float().to(device)  # Fix wrong voxel offset

            flat_features = sample_features.view(num_points * num_repetition, -1)
            sparse_coordinates.append(flatten_coordinates)
            sparse_features.append(flat_features)

        if len(sparse_coordinates) == 0:
            return None

        # batch
        sparse_features = torch.cat(sparse_features, dim=0)
        batched_coordinates = Me.utils.batched_coordinates(sparse_coordinates).to(device)
        tensor = Me.SparseTensor(features=sparse_features,
                                 coordinates=batched_coordinates,
                                 quantization_mode=Me.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)

        return tensor

    def compute_frustum_padding(self, intrinsic_inverse: torch.Tensor) -> torch.Tensor:
        frustum = generate_frustum(self.image_size, intrinsic_inverse.cpu(), self.depth_min, self.depth_max)
        dimensions, _ = generate_frustum_volume(frustum, config.MODEL.PROJECTION.VOXEL_SIZE)
        difference = (self.frustum_dimensions - torch.tensor(dimensions)).float().to(intrinsic_inverse.device)

        padding_offsets = difference // 2

        return padding_offsets

    def inference(self, depth, features, instances, intrinsic) -> Me.SparseTensor:
        data = FieldList(depth.shape[2:])
        data.add_field("depth", DepthMap(depth, intrinsic))

        return self.forward(depth, features, instances, [data])