import sys
sys.path.append("..")

from types import SimpleNamespace
from utils.point_transformer_loader import get_data_loader
from utils.visualizers import lidar_to_bev

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# Configure the data loader

config = {
    "data": "data/AutoCast_6/Train",
    "daggerdata": "data/AutoCast_6/Dagger",
    "ego_only": False,
    "batch_size": 1,
    "num_dataloader_workers": 1,
    "use_lidar": True,
    "visualize": True,
    "shared": True,
    "earlyfusion": False,
    "max_num_neighbors": 2,
    "npoints": 4096,
    "uniform": True,
    "z_min": -4,
    "z_max": 12,
}

config = SimpleNamespace(**config)
data_loader = get_data_loader(config)

NUM = 2

bev_rgb, \
ego_lidar, ego_speed, ego_brake, ego_has_plan, ego_command, ego_control, \
other_lidar, other_speed, other_transform, ego_transform, num_valid_neighbors = next(iter(data_loader))

pcd = o3d.geometry.PointCloud()
print(ego_lidar.shape)
pcd.points = o3d.utility.Vector3dVector(ego_lidar[0])

o3d.visualization.draw_geometries([pcd])