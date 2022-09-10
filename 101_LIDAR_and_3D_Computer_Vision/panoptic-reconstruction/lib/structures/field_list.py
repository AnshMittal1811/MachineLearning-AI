# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# transpose
from typing import List

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class FieldList:
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, image_size, mode="xyxy"):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def update(self, dictionary):
        for k, v in dictionary.items():
            self.extra_fields[k] = v

    def copy_with_fields(self, fields, skip_missing=False):
        field_list = FieldList(self.size, self.mode)

        if not isinstance(fields, (list, tuple)):
            fields = [fields]

        for field in fields:
            if self.has_field(field):
                field_list.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))

        return field_list

    def __len__(self):
        return len(self.extra_fields)

    def __getitem__(self, item):
        field_list = FieldList(self.size, self.mode)

        for k, v in self.extra_fields.items():
            field_list.add_field(k, v[item])

        return field_list

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)

        return s


def collect(data: List[FieldList], field: str, device: str = "cuda", access_fn=None) -> torch.Tensor:
    if access_fn is None:
        return torch.stack([conditional_to(t.get_field(field), device) for t in data], dim=0)
    else:
        return torch.stack([conditional_to(access_fn(t.get_field(field)), device) for t in data], dim=0)


def conditional_to(x: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    if device is None:
        return x
    else:
        return x.to(device, non_blocking=True)
