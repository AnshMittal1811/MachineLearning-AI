from datetime import datetime
from pathlib import Path
from typing import List, Any, Union, Tuple, Dict

import numpy as np

import torch
from torch.nn import functional as F

from scipy import ndimage
from scipy.spatial.transform.rotation import Rotation as R


class Compose:
    def __init__(self, transforms: List[Any], profiling: bool = False) -> None:
        self.transforms = transforms
        self.profiling = profiling

    def __call__(self, data, *args, **kwargs):
        if self.profiling:
            data, timings = self.call_with_profiling(data, *args, **kwargs)
            return data, timings
        else:
            for transform in self.transforms:
                data = transform(data, *args, **kwargs)

            return data

    def call_with_profiling(self, data, *args, **kwargs):
        timings = {}

        total_start = datetime.now()

        for transform in self.transforms:
            start = datetime.now()
            data = transform(data, *args, **kwargs)
            end = datetime.now()

            name = type(transform).__name__
            timings[name] = (end - start).total_seconds()

        total_end = datetime.now()
        name = type(self).__name__
        timings[name] = (total_end - total_start).total_seconds()

        return data, timings


class FromNumpy:
    def __call__(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        tensor = torch.from_numpy(data)
        return tensor


class ToTensor:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.dtype is None:
            tensor = torch.as_tensor(data)
        else:
            tensor = torch.as_tensor(data, dtype=self.dtype)
        return tensor


class ToOccupancy:
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def __call__(self, distance_field: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        one = torch.ones([1], device=distance_field.device)
        zero = torch.zeros([1], device=distance_field.device)
        occupancy_grid = torch.where(torch.abs(distance_field) < self.threshold, one, zero)
        return occupancy_grid


class ToTDF:
    def __init__(self, truncation):
        self.truncation = truncation

    def __call__(self, distance_field: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        distance_field = torch.abs(distance_field)
        distance_field = torch.clip(distance_field, 0, self.truncation)
        return distance_field


class ToTSDF:
    def __init__(self, truncation):
        self.truncation = truncation

    def __call__(self, distance_field: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        distance_field = torch.clip(distance_field, -self.truncation, self.truncation)
        return distance_field


class ToBinaryMask:
    def __init__(self, threshold: float, compare_function=torch.lt):
        self.threshold = threshold
        self.compare_function = compare_function

    def __call__(self, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        mask = self.compare_function(mask, self.threshold)
        return mask


class Absolute:
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.abs(x)


class NormalizeOccupancyGrid:
    def __call__(self, occupancy_grid: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        normalized = (occupancy_grid * 2) - 1
        return normalized


class Flip:
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        flipped = torch.flip(volume, self.dims)

        return flipped


class Transpose:
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        transposed = torch.transpose(volume, self.dims[0], self.dims[1])

        return transposed


class Pad:
    def __init__(self, target_size: Union[List, torch.Tensor], padding_value: float = 0.0, ignore_features=False):
        self.target_size = target_size
        self.padding_value = padding_value
        self.ignore_features = ignore_features

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        dimensions = torch.Size(self.target_size)

        diff = compute_dimension_difference(dimensions, volume)
        diff_a = torch.floor(diff / 2)
        diff_b = torch.ceil(diff / 2)
        padding = [item.int().item() for sublist in zip(diff_a, diff_b) for item in sublist]
        padding_back_to_front = padding[::-1]

        padded_result = F.pad(volume, padding_back_to_front, value=self.padding_value, mode="constant")

        return padded_result


class Crop:
    def __init__(self, max_dimensions: Union[List, torch.Tensor], ignore_features=False):
        self.max_dimensions = max_dimensions
        self.ignore_features = ignore_features

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        dimensions = torch.Size(self.max_dimensions)

        diff = compute_dimension_difference(dimensions, volume)
        diff_left = torch.floor(diff / 2).int().tolist()
        diff_right = torch.ceil(diff / 2).int().tolist()

        # disable negative cropping (padding)
        diff_left = [0 if d > 0 else -d for d in diff_left]
        diff_right = [None if d >= 0 else d for d in diff_right]

        if volume.dim() == 5:
            cropped = volume[:, :, diff_left[0]:diff_right[0], diff_left[1]:diff_right[1], diff_left[2]:diff_right[2]]
        elif volume.dim() == 4:
            cropped = volume[:, diff_left[0]:diff_right[0], diff_left[1]:diff_right[1], diff_left[2]:diff_right[2]]
        elif volume.dim() == 3:
            cropped = volume[diff_left[0]:diff_right[0], diff_left[1]:diff_right[1], diff_left[2]:diff_right[2]]
        else:
            cropped = volume
        return cropped


def compute_dimension_difference(dimensions: torch.Size, volume: torch.Tensor) -> torch.Tensor:
    source_dimensions = volume.dim()
    if source_dimensions == 5:  # B,C,XYZ
        difference = torch.tensor(dimensions) - torch.tensor(volume.shape[2:])
    elif source_dimensions == 4:  # B,XYZ
        difference = torch.tensor(dimensions) - torch.tensor(volume.shape[1:])
    elif source_dimensions == 3:  # XYZ
        difference = torch.tensor(dimensions) - torch.tensor(volume.shape)
    else:
        difference = torch.zeros(len(dimensions))
    return difference.float()


class Resize:
    def __init__(self, target_size: Union[List, torch.Tensor], mode: str = "trilinear"):
        self.target_size = target_size
        self.mode = mode

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        while len(volume.size()) < 5:
            volume = volume.unsqueeze(0)

        resized = F.interpolate(volume, size=self.target_size, mode=self.mode, align_corners=False)

        return resized.squeeze(0)


class ResizeTrilinear:
    def __init__(self, factor: float, mode: str = "trilinear"):
        self.factor = factor
        self.mode = mode
        self.mode_args = {
            "trilinear": {
                "recompute_scale_factor": False,
                "align_corners": True
            },
            "nearest": {
                "recompute_scale_factor": True
            }
        }

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        while len(volume.size()) < 5:
            volume = volume.unsqueeze(0)

        mode_args = self.mode_args.get(self.mode, {})
        resized = F.interpolate(volume, scale_factor=self.factor, mode=self.mode, **mode_args)

        return resized.squeeze(0)


class ResizeMax:
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        old_dtype = volume.type()
        old_dim = volume.dim()

        while volume.dim() < 5:
            volume.unsqueeze_(0)

        volume = volume.type(torch.float)

        resized = F.max_pool3d(volume, self.kernel_size, self.stride, self.padding)
        resized = resized.type(old_dtype)

        while resized.dim() > old_dim:
            resized.squeeze_(0)

        return resized


class ResizeBy:
    def __init__(self, factor: float, mode: str = "trilinear"):
        self.factor = factor
        self.mode = mode
        self.mode_args = {
            "trilinear": {
                "recompute_scale_factor": False,
                "align_corners": True
            },
            "nearest": {
                "recompute_scale_factor": True
            }
        }

    def __call__(self, volume: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        while len(volume.size()) < 5:
            volume = volume.unsqueeze(0)

        if self.mode == "max":
            resized = F.max_pool3d(volume, 2, 2, 0)
        else:
            mode_args = self.mode_args.get(self.mode, {})

            resized = F.interpolate(volume, scale_factor=self.factor, mode=self.mode, **mode_args)

        return resized.squeeze(0)


class Mask:
    def __init__(self, mask: torch.Tensor, value: float) -> None:
        self.mask = mask
        self.value = value

    def __call__(self, volume: torch.Tensor, value: float = None, *args, **kwargs) -> torch.Tensor:
        fill_value = self.value

        if value is not None and isinstance(value, float):
            fill_value = value

        masked = torch.masked_fill(volume, self.mask, fill_value)
        return masked


class MaskByField:
    def __init__(self, field: str, fill_value: float, threshold: float):
        self.field = field
        self.fill_value = fill_value
        self.threshold = threshold

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # get data from field
        # field_data = args[0][1][self.field]
        field_data = args[0][0].data[self.field]

        # convert to mask
        mask = field_data > self.threshold
        mask = torch.from_numpy(mask)

        # mask input
        masked = torch.masked_fill(x, mask, self.fill_value)
        return masked


class Sparsify:
    def __call__(self, occupancy: torch.Tensor, features=None, *args, **kwargs) -> Tuple[np.array, np.array, np.array]:
        device = occupancy.device
        ones = torch.ones([1], device=device)
        zeros = torch.zeros([1], device=device)
        coords = torch.stack(torch.where(occupancy.squeeze(1) == 1.0, ones, zeros).nonzero(as_tuple=True), dim=1).int()

        if features is not None and len(features.shape) == len(occupancy.shape):
            num_dimensions = coords.shape[1]
            locations = coords.long()
            if num_dimensions == 4:  # BxCx Volume
                feats = features[locations[:, 0], :, locations[:, 1], locations[:, 2], locations[:, 3]]
            elif num_dimensions == 3:
                feats = features[0, :, locations[:, 0], locations[:, 1], locations[:, 2]]
                feats = feats.permute(1, 0)
            else:
                feats = torch.ones_like(coords[:, :1], dtype=torch.float)
        else:
            feats = torch.ones_like(coords[:, :1], dtype=torch.float)

        labels = torch.ones_like(coords[:, :1], dtype=torch.int32)

        return coords, feats, labels


class Normalize:
    def __init__(self, range_input: Tuple[float, float], range_output: Tuple[float, float]):
        self.range_input = range_input
        self.range_output = range_output

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        scale = self.range_output[1] - self.range_output[0]
        denom = self.range_input[1] / scale
        normalized = (x / denom) + self.range_output[0]

        return normalized

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.range_output[1] - self.range_output[0]
        denom = self.range_input[1] / scale
        unnormalized = (x - self.range_output[0]) * denom
        return unnormalized


class NormalizeByStats:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        normalized = (x - self.mean) / self.std

        return normalized

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        unnormalized = (x * self.std) + self.mean
        return unnormalized


class Unsqueeze:
    def __init__(self, dimension: int = 0):
        self.dimension = dimension

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x.unsqueeze(self.dimension)


class Mapping:
    def __init__(self, mapping: Dict, default_value: int = 0, ignore_values: List = None) -> None:
        self.mapping = mapping
        self.default_value = default_value

        if ignore_values is None:
            ignore_values = []

        self.ignore_values = ignore_values

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        mapped = torch.full_like(x, self.default_value)
        uniques = torch.unique(x)

        # Use passed mapping if present
        mapping = kwargs.get("mapping", self.mapping)

        for unique in uniques.tolist():
            if unique in self.ignore_values:
                continue

            mask = x == unique
            mapped[mask] = mapping.get(unique, self.default_value)
        return mapped


class OneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, semantics: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        one_hot = F.one_hot(semantics.long().squeeze(0), self.num_classes).permute(3, 0, 1, 2)
        return one_hot


class NoOp:
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
