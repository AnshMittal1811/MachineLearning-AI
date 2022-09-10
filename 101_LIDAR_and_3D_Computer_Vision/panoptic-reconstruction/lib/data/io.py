import os
import struct
from typing import Tuple, List

import numpy as np

TYPE_NAMES = {
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float": "f",
    "double": "d",
    "char": "s"
}


class BinaryReader:
    def __init__(self, file_path):
        self.file = open(file_path, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def read(self, type_name, times=1):
        type_format = TYPE_NAMES[type_name.lower()] * times
        type_size = struct.calcsize(type_format)
        value = self.file.read(type_size)
        if type_size != len(value):
            print(f"Error while parsing {self.file}")
            raise IOError(f"Error while parsing {self.file}")
        return struct.unpack(type_format, value)

    def close(self):
        self.file.close()


class BinaryWriter(object):
    def __init__(self, file_path):
        self.file = open(file_path, "wb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def write(self, type_name, value):
        type_format = TYPE_NAMES[type_name.lower()]
        self.file.write(struct.pack(f"={value.size}{type_format}", *value.flatten("F")))

    def write_bytes(self, bytes_data):
        self.file.write(bytes_data)

    def close(self):
        self.file.close()


def read_dense_distance_field(file_path: os.PathLike) -> np.array:
    with BinaryReader(file_path) as reader:
        dim_x, dim_y, dim_z = reader.read("unit64", 3)
        grid = reader.read("float", dim_x * dim_y * dim_z)
        grid = np.array(grid, dtype=np.float32).reshape([dim_x, dim_y, dim_z], order="F")

    return grid


def read_spare_distance_field(file_path: os.PathLike) -> Tuple[List, np.array, np.array]:
    with BinaryReader(file_path) as reader:
        dim_x, dim_y, dim_z = reader.read("uint64", 3)
        num_voxels = reader.read("uint64", 1)[0]
        locations = reader.read("uint32", 3 * num_voxels)
        locations = np.asarray(locations).reshape([num_voxels, 3]).astype(np.int)
        values = reader.read("float", num_voxels)

    return [dim_x, dim_y, dim_z], locations, values


def read_sparse_distance_field_to_dense(file_path: os.PathLike, default_value: float = 0.0) -> np.array:
    (dim_x, dim_y, dim_z), locations, values = read_spare_distance_field(file_path)

    grid = np.full((dim_x, dim_y, dim_z), fill_value=default_value, dtype=np.float)
    grid[locations[:, 0], locations[:, 1], locations[:, 2]] = values

    return grid


def read_dense_segmentation(file_path: os.PathLike, offset_value: int = 1000) -> Tuple[np.array, np.array]:
    with BinaryReader(file_path) as reader:
        dim_x, dim_y, dim_z = reader.read("uint64", 3)
        segmentation = reader.read("uint32", dim_x * dim_y * dim_z)

    segmentation = np.array(segmentation, dtype=np.uint32).reshape([dim_x, dim_y, dim_z], order="F")

    semantic = segmentation // offset_value
    instance = segmentation % offset_value

    return semantic, instance


def read_spare_segmentation(file_path: os.PathLike, offset_value: int = 1000) -> Tuple[List, np.array, np.array, np.array]:
    with BinaryReader(file_path) as reader:
        dim_x, dim_y, dim_z = reader.read("uint64", 3)
        num_voxels = reader.read("uint64", 1)[0]
        locations = reader.read("uint32", 3 * num_voxels)
        locations = np.asarray(locations).reshape([num_voxels, 3]).astype(int)
        segmentation = reader.read("uint32", num_voxels)

    segmentation = np.array(segmentation, dtype=np.uint32)

    semantic = segmentation // offset_value
    instance = segmentation % offset_value

    return [dim_x, dim_y, dim_z], locations, semantic, instance


def read_sparse_segmentation_to_dense(file_path: os.PathLike, offset_value: int = 1000,
                                      default_value: int = 0) -> Tuple[np.array, np.array]:
    [dim_x, dim_y, dim_z], locations, semantic, instance = read_spare_segmentation(file_path, offset_value)

    semantic_grid = np.full_like((dim_x, dim_y, dim_z), dtype=np.uint32, fill_value=default_value)
    semantic_grid[locations[:, 0], locations[:, 1], locations[:, 2]] = semantic

    instance_grid = np.full_like((dim_x, dim_y, dim_z), dtype=np.uint32, fill_value=default_value)
    instance_grid[locations[:, 0], locations[:, 1], locations[:, 2]] = instance

    return semantic_grid, instance_grid


def assemble_frame_name(frame_name, type_name: str, extension: str, drop_yaw: bool = False):
    frame_parts = frame_name.split("_")
    frame_name = frame_parts[0]
    frame_angle = frame_parts[1]
    frame_rot = frame_parts[2]

    if drop_yaw:
        file_name = f"{frame_name}_{type_name}{frame_angle}{extension}"
    else:
        file_name = f"{frame_name}_{type_name}{frame_angle}_{frame_rot}{extension}"

    return file_name
