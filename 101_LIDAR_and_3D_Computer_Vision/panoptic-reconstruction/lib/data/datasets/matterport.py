import os
import random
from pathlib import Path
from typing import Dict, Union, List, Tuple

import numpy as np
import torch.utils.data
from PIL import Image

from torchvision.transforms import ColorJitter

from lib import data
from lib.data import io
from lib.data import transforms2d as t2d
from lib.data import transforms3d as t3d
from lib.structures import FieldList
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class Matterport(torch.utils.data.Dataset):
    def __init__(self, file_list_path: os.PathLike, dataset_root_path: os.PathLike, fields: List[str],
                 num_samples: int = None, shuffle: bool = False) -> None:
        super().__init__()

        self.dataset_root_path = Path(dataset_root_path)

        self.samples: List = self.load_and_filter_file_list(file_list_path)

        if shuffle:
            random.shuffle(self.samples)

        self.samples = self.samples[:num_samples]

        # Fields defines which data should be loaded
        if fields is None:
            fields = []

        self.fields = fields

        # TODO
        # - filter samples with too few voxels?

        self.image_size = (320, 240)
        self.depth_image_size = (160, 120)
        self.voxel_size = config.MODEL.PROJECTION.VOXEL_SIZE
        self.depth_min = config.MODEL.PROJECTION.DEPTH_MIN
        self.depth_max = config.MODEL.PROJECTION.DEPTH_MAX
        self.grid_dimensions = config.MODEL.FRUSTUM3D.GRID_DIMENSIONS
        self.truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        self.max_instances = config.MODEL.INSTANCE2D.MAX
        self.num_min_instance_pixels = config.MODEL.INSTANCE2D.MIN_PIXELS
        self.stuff_classes = [0, 10, 11, 12]

        self.transforms: Dict = self.define_transformations()

    def __getitem__(self, index) -> Tuple[str, FieldList]:
        sample_path = self.samples[index]
        scene_id = sample_path.split("/")[0]
        image_id = sample_path.split("/")[1]

        sample = FieldList(self.image_size, mode="xyxy")
        sample.add_field("index", index)
        sample.add_field("name", sample_path)

        # 2D data
        if "color" in self.fields:
            color = Image.open(self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "i", ".jpg"), formats=["JPEG"])
            color = self.transforms["color"](color)
            sample.add_field("color", color)

        if "depth" in self.fields:
            depth = Image.open(self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "d", ".png"), formats=["PNG"])
            depth = self.transforms["depth"](depth)

            # divide by depth unit
            depth.depth_map /= 4000

            # mask depth pixels outside the frustum (too close/too far)
            depth.depth_map[depth.depth_map > self.depth_max] = 0
            depth.depth_map[depth.depth_map < self.depth_min] = 0

            # load intrinsic
            intrinsic_path = self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "intrinsics_", ".npy", drop_yaw=True)
            intrinsic = np.load(intrinsic_path).reshape(3, 3)

            intrinsic = adjust_intrinsic(intrinsic, self.image_size, self.depth_map_size)
            intrinsic = torch.from_numpy(intrinsic).float()
            depth.intrinsic_matrix = intrinsic

            sample.add_field("depth", depth)

        if "instance2d" in self.fields:
            segmentation2d = np.load(self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "segmap", ".npz"))["data"]
            instance2d = self.transforms["instance2d"](segmentation2d)
            sample.add_field("instance2d", instance2d)

            room_mask = segmentation2d[..., 2]
            sample.add_field("room_mask", room_mask)

        # 3D data
        needs_weighting = False
        if "geometry" in self.fields:
            geometry_path = self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "geometry", ".npz")
            geometry_content = np.load(geometry_path)
            geometry = self.transforms["geometry"](geometry_content["data"])

            mask = self.transforms["mask"](geometry_content["mask"])

            # process hierarchy
            sample.add_field("occupancy_256", self.transforms["occupancy_256"](geometry.abs()))
            sample.add_field("occupancy_128", self.transforms["occupancy_128"](geometry.abs()))
            sample.add_field("occupancy_64", self.transforms["occupancy_64"](geometry.abs()))

            geometry = self.transforms["geometry_truncate"](geometry)
            sample.add_field("geometry", geometry)

            # add frustum mask
            sample.add_field("frustum_mask", mask)

            needs_weighting = True

        if "semantic3d" or "instance3d" in self.fields:
            segmentation3d_path = self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "sem", ".npz")
            semantic3d, instance3d = np.load(segmentation3d_path)["data"]
            needs_weighting = True

            if "semantic3d" in self.fields:
                semantic3d = self.transforms["semantic3d"](semantic3d)
                sample.add_field("semantic3d", semantic3d)

                # process semantic3d hierarchy
                sample.add_field("semantic3d_64", self.transforms["segmentation3d_64"](semantic3d))
                sample.add_field("semantic3d_128", self.transforms["segmentation3d_128"](semantic3d))

            if "instance3d" in self.fields:
                # Ensure consistent instance id shuffle between 2D and 3D instances
                instance_mapping = sample.get_field("instance2d").get_field("instance_mapping")
                instance3d = self.transforms["instance3d"](instance3d, mapping=instance_mapping)
                sample.add_field("instance3d", instance3d)

                # process instance3d hierarchy
                sample.add_field("instance3d_64", self.transforms["segmentation3d_64"](instance3d))
                sample.add_field("instance3d_128", self.transforms["segmentation3d_128"](instance3d))

        if needs_weighting:
            weighting_path = self.dataset_root_path / scene_id / f"weighting_{image_id}.npz"
            weighting = np.load(weighting_path)["data"]
            weighting = self.transforms["weighting3d"](weighting)
            sample.add_field("weighting3d", weighting)

            # Process weighting mask hierarchy
            sample.add_field("weighting3d_64", self.transforms["weighting3d_64"](weighting))
            sample.add_field("weighting3d_128", self.transforms["weighting3d_128"](weighting))

        return sample_path, sample

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_and_filter_file_list(file_list_path: os.PathLike) -> List[str]:
        with open(file_list_path) as f:
            content = f.readlines()

        images = [line.strip() for line in content]

        return images

    def define_transformations(self) -> Dict:
        transforms = dict()

        # 2D transforms
        transforms["color"] = t2d.Compose([
            t2d.ToTensor(),
            ColorJitter(0.4, 0.4, 0.4, 0.4),
            t2d.Normalize(_imagenet_stats["mean"], _imagenet_stats["std"])
        ])

        transforms["depth"] = t2d.Compose([
            t2d.ToImage(),
            t2d.Resize(self.depth_image_size, Image.NEAREST),
            t2d.ToNumpyArray(),
            t2d.ToTensorFromNumpy(),
            t2d.ToDepthMap(None)   # Set image specific intrinsic later
        ])

        transforms["instance2d"] = t2d.Compose([
            t2d.SegmentationToMasks(self.image_size, self.num_min_instance_pixels,
                                    self.max_instances, True, self.stuff_classes)
        ])

        # 3D transforms
        transforms["geometry"] = t3d.Compose([
            t3d.ToTensor(dtype=torch.float),
            t3d.Unsqueeze(0),
            t3d.ToTSDF(truncation=12)
        ])

        transforms["mask"] = t3d.Compose([
            t3d.ToTensor(dtype=torch.bool),
        ])

        transforms["geometry_truncate"] = t3d.ToTSDF(truncation=self.truncation)

        transforms["occupancy_64"] = t3d.Compose([t3d.Absolute(), t3d.ResizeTrilinear(0.25),
                                                  t3d.ToBinaryMask(8),
                                                  t3d.ToTensor(dtype=torch.float)])

        transforms["occupancy_128"] = t3d.Compose([t3d.Absolute(), t3d.ResizeTrilinear(0.5),
                                                   t3d.ToBinaryMask(6),
                                                   t3d.ToTensor(dtype=torch.float)])

        transforms["occupancy_256"] = t3d.Compose([t3d.Absolute(),
                                                   t3d.ToBinaryMask(self.truncation),
                                                   t3d.ToTensor(dtype=torch.float)])

        transforms["weighting3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.float), t3d.Unsqueeze(0)])
        transforms["weighting3d_64"] = t3d.ResizeTrilinear(0.25)
        transforms["weighting3d_128"] = t3d.ResizeTrilinear(0.5)

        transforms["semantic3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long)])

        transforms["instance3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long), t3d.Mapping(mapping={}, ignore_values=[0])])

        transforms["segmentation3d_64"] = t3d.Compose([t3d.ResizeMax(8, 4, 2)])
        transforms["segmentation3d_128"] = t3d.Compose([t3d.ResizeMax(4, 2, 1)])

        return transforms
