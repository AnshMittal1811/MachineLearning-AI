"""
PCD I/O tools.
"""

from pathlib import Path

import numpy as np
import open3d as o3d

import utils.math_utils as math_utils


def load_point_cloud(pcd_file: Path):
    """
    Loads single pointcloud.
    """
    meta_file = pcd_file.parent / (pcd_file.stem + '_meta.npy')
    meta = None
    if meta_file.is_file():
        meta = np.load(meta_file, allow_pickle=True).item()

    # Load pointcloud.
    if pcd_file.suffix in ['.xyz', '.xyzn']:
        points, normals, colors = _load_xyzn(pcd_file)
    else:
        points, normals, colors = _load_open3d(pcd_file)

    # Defaults.
    if normals is None or normals.shape[0] < points.shape[0]:
        normals = np.zeros_like(points)
    if colors is None or colors.shape[0] < points.shape[0]:
        colors = np.zeros_like(points)

    # Covert to world coordiantes.
    m_pcd_to_original = np.eye(4, dtype=np.float32)
    if meta is not None:
        is_in_world_coords = meta.get('is_in_world_coords', False)
        if not is_in_world_coords and 'view' in meta:
            # Transform from camera to world space.
            cam_to_world = np.linalg.inv(meta['view'])
            points = math_utils.transform_points(cam_to_world, points, return_euclidean=True)
            normals = math_utils.transform_normals(cam_to_world, normals)

        # Remember pcd->original transform.
        if 'm_pcd_to_original' in meta:
            m_pcd_to_original = meta['m_pcd_to_original']

    return points, normals, colors, m_pcd_to_original


def load_all_point_clouds(dataset_path: Path):
    """
    Loads all PCDs in given path.
    """
    pcd_files = sorted([x for x in dataset_path.iterdir() if x.suffix in ['.ply', '.xyz', '.xyzn']])
    if len(pcd_files) == 0:
        # No PCD.
        return None

    all_points = []
    all_normals = []
    all_colors = []
    offsets = [0]
    m_pcd_to_original = np.eye(4, dtype=np.float32)
    for pcd_file in pcd_files:
        # Has metadata?
        print(f'\tLoading PCD from {pcd_file}.')
        points, normals, colors, m_pcd_to_original = load_point_cloud(pcd_file)

        all_points += [points]
        all_normals += [normals]
        all_colors += [colors]
        offsets += [offsets[-1] + points.shape[0]]

    # Merge all pcds.
    return {
        'points': np.concatenate(all_points, 0),
        'normals': np.concatenate(all_normals, 0),
        'colors': np.concatenate(all_colors, 0),
        'offsets': np.array(offsets, int),
        'm_pcd_to_original': m_pcd_to_original,
    }


def _load_xyzn(pointcloud_path: Path):
    """
    Loads coords and normals from xyzn text file.
    """
    print("Loading XYZN point cloud")
    pointcloud_path_bin = Path(str(pointcloud_path) + ".npy")
    if pointcloud_path_bin.is_file():
        # Use binary cache for faster loading.
        point_cloud = np.load(pointcloud_path_bin)
    else:
        # Do slow text parsing.
        point_cloud = np.genfromtxt(pointcloud_path)
        # Generate binary cache for faster loading.
        np.save(pointcloud_path_bin, point_cloud)
    print("Finished XYZN loading point cloud")
    return point_cloud[:, :3], point_cloud[:, 3:], None


def _load_open3d(pointcloud_path: Path):
    """
    Loads coords and normals and colors from PLY or PCD file using Open3D.
    """
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    coords = np.array(pcd.points)
    normals = np.array(pcd.normals)
    colors = np.array(pcd.colors)
    return coords, normals, colors
