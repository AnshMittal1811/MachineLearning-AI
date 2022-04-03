"""
PCD conversion tools.
"""

import argparse
from pathlib import Path

import numpy as np
import imageio
import open3d as o3d
import cv2

import matplotlib.pyplot as plt
import utils.math_utils as math_utils


def intrinsics_to_gl_frustrum(intrinsics: np.array, resolution: np.array, znear=1, zfar=1e3):
    """
    Computes OpenGL projection frustrum
    equivalent to a OpenCV intrinsics.
    https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
    """
    width, height = resolution
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, -1]
    cy = intrinsics[1, -1]

    m = np.eye(4, dtype=np.float32)
    m[0][0] = 2.0 * fx / width
    m[0][1] = 0.0
    m[0][2] = 0.0
    m[0][3] = 0.0

    m[1][0] = 0.0
    m[1][1] = 2.0 * fy / height
    m[1][2] = 0.0
    m[1][3] = 0.0

    m[2][0] = 1.0 - 2.0 * cx / width
    m[2][1] = -(1.0 - 2.0 * cy / height)
    m[2][2] = (zfar + znear) / (znear - zfar)
    m[2][3] = -1.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 2.0 * zfar * znear / (znear - zfar)
    m[3][3] = 0.0

    return m.T


def gl_frustrum_to_intrinsics(m_projection: np.array, resolution: np.array):
    """
    Computes OpenCV intrinsics from OpenGL projection frustrum.
    https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
    """
    width, height = resolution

    fx = m_projection[0, 0] * width / 2.0
    fy = m_projection[1, 1] * height / 2.0
    cx = (1.0 - m_projection[0, 2]) * width / 2.0
    cy = (1.0 + m_projection[1, 2]) * height / 2.0

    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, -1] = cx
    intrinsics[1, -1] = cy
    return intrinsics


def undistort_image(img, camera_matrix: np.array, dist_coeff: np.array, keep_borders=False):
    """
    Undistort the image, if the loaded matrix and distortion parameters have been given.
    """
    new_mtx = camera_matrix
    if not keep_borders:
        new_mtx = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (img.shape[1], img.shape[0]), 0)[0]
    return cv2.undistort(img, camera_matrix, dist_coeff, newCameraMatrix=new_mtx), new_mtx


def pcd_to_mesh(pcd):
    """
    Pointcloud ot mesh.
    """
    # estimate radius for rolling ball
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2]))
    return mesh


def valid_coords(coords, resolution):
    """
    Gets validity mask for 2d image px coords.
    """
    return np.logical_and(
        np.logical_and(coords[..., 0] >= 0, coords[..., 0] < resolution[0]),
        np.logical_and(coords[..., 1] >= 0, coords[..., 1] < resolution[1]))


def resample_image(image, coords):
    """
    Samples image values.
    """
    batch_size = 32767 - 1
    num_batches = int(np.ceil(coords.shape[0] / batch_size))
    samples = []
    for i in range(num_batches):
        points_batch = coords[i * batch_size:(i + 1) * batch_size, ...]
        samples += [cv2.remap(image, points_batch[None, ...].astype(np.float32),
                              None, interpolation=cv2.INTER_NEAREST)[0]]
    return np.concatenate(samples, 0)


def render_pcd(pcd, view_matrix, projection_matrix, resolution):
    """
    Renders PCD under projection.
    """
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)

    # Project points.
    points_2d = math_utils.transform_points(
        projection_matrix @ view_matrix, points, return_euclidean=True)
    #coords_2d = ((points_2d[..., :2] + 1) / 2 * (resolution - 1) + 0.5).astype(int)
    coords_2d = ((points_2d[..., :2] + 1) / 2 * resolution).astype(int)

    # V flip
    coords_2d[..., 1] = resolution[1] - 1 - coords_2d[..., 1]

    # Remove out of canvas.
    mask = valid_coords(coords_2d, resolution)
    coords_2d = coords_2d[mask, :]
    colors = colors[mask, :]

    # Paint.
    canvas = np.zeros((resolution[1], resolution[0], 3), np.float32)
    canvas[coords_2d[:, 1], coords_2d[:, 0], :] = colors
    return canvas, coords_2d, colors


def filter_reliable_normals(pcd_cam, look_dir=np.array([0, 0, -1], dtype=np.float32)):
    """
    Returns normal validity mask.
    Removes normals that are too orthogonal to camera.
    Flips normals that are parallel but wrong way.
    """
    # Compute angle to camera view.
    look_dir /= np.linalg.norm(look_dir)
    normals = np.array(pcd_cam.normals)
    angle_deg = np.degrees(np.arccos(np.dot(normals, look_dir)))

    # # Flip flipped.
    # is_flipped = angle_deg < 90
    # normals[is_flipped, :] *= -1
    # angle_deg[is_flipped] = 180 - angle_deg[is_flipped]

    # Remove orthogonal.
    is_valid = angle_deg > 155
    normals[np.logical_not(is_valid), :] = 0

    pcd_cam.normals = o3d.utility.Vector3dVector(normals)
    return pcd_cam, is_valid


def estimate_normals(pcd_world, view_matrix):
    """
    Estimates normals.
    """
    # Convert to camera space.
    points_world = np.array(pcd_world.points)
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(
        math_utils.transform_points(view_matrix, points_world, return_euclidean=True))

    # Estimate normals.
    pcd_cam.estimate_normals()
    pcd_cam, is_normal_valid = filter_reliable_normals(pcd_cam)
    #o3d.visualization.draw_geometries([pcd_cam.uniform_down_sample(every_k_points=20)], point_show_normal=True)

    # Project back to world space.
    normals_world = math_utils.transform_normals(np.linalg.inv(view_matrix), np.array(pcd_cam.normals))
    pcd_world.normals = o3d.utility.Vector3dVector(normals_world)
    #o3d.visualization.draw_geometries([pcd_world.uniform_down_sample(every_k_points=20)], point_show_normal=True)
    return pcd_world


def display_inlier_outlier(pcd, inliers_inds):
    """
    Colorcodes inliers and outliers.
    """
    inlier_cloud = pcd.select_by_index(inliers_inds)
    outlier_cloud = pcd.select_by_index(inliers_inds, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def normalize_pcd_poses(pcds_with_metas, taret_reference_view: np.array):
    """
    Normalizes PCD poses so that the reference view is as desired.
    """
    _, meta_ref = pcds_with_metas[len(pcds_with_metas) // 2]
    m_transform = np.linalg.inv(meta_ref['view']) @ taret_reference_view
    for pcd, meta in pcds_with_metas:
        # Update view.
        meta['view'] = meta['view'] @ m_transform

        if pcd is not None:
            # Update points (inverse).
            points = np.array(pcd.points)
            points_new = math_utils.transform_points(np.linalg.inv(m_transform), points, return_euclidean=True)
            pcd.points = o3d.utility.Vector3dVector(points_new)

    return m_transform


def imwritef(filename, im):
    """
    Saves float image.
    """
    imageio.imwrite(filename, (np.clip(im, 0, 1) * 255).astype(np.uint8))


def validate_pcd_projection(im, im_meta, pcd, pcd_meta, output_path: Path = None):
    """
    Projects pcd to image and save to drive.
    """

    # Render PCD.
    resolution = np.array([im.shape[1], im.shape[0]], int)
    render, render_coords, render_colors = render_pcd(pcd, im_meta['view'], im_meta['projection'], resolution)

    # Write.
    if output_path is not None:
        out_dir = output_path / 'viz'
        out_dir.mkdir(0o777, True, True)
        imwritef(out_dir / f'{im_meta["name"]}_from_{pcd_meta["name"]}_render.png', render)
        imwritef(out_dir / f'{im_meta["name"]}_gt.png', im)

    # Compute error.
    gt = im[render_coords[:, 1], render_coords[:, 0], :]
    error = ((render_colors - gt) ** 2).mean()
    print(f'Projection of {pcd_meta["name"]} to {im_meta["name"]} => Error = {error:.3f}')
    return error


def find_lines_intersection_3D(origins, directions, min_det=1e-5, interp_factor=0.5):
    """
    Least squares 3D line intersection approximation.
    If the lines are too parallel, then use the interp_factor to interpolate the
    near and far bounds in diopter space.
    Origins and directions are Nx3 arrays.
    """

    A = np.zeros((3, 3))
    b = np.zeros(3)

    # Normalize directions.
    directions = (directions.T / np.linalg.norm(directions, axis=1)).T

    for i, origin_i in enumerate(origins):

        # uu_i = Dir_i dot Dir_i^T
        uu = np.outer(directions[i], directions[i])
        # Iuu_i = I_3x3 - uu_i
        Iuu = np.eye(3) - uu
        # A = Sum  Iuu_i
        A += Iuu
        # b = Sum_i  Iuu_i dot origin_i
        b += Iuu @ origin_i

    if np.linalg.det(A) < min_det:
        raise RuntimeError("Could determine the lines intersection.")
    else:

        # Solve x from Ax = b.
        return np.linalg.inv(A) @ b


def downsample_pcd(pcd: o3d.geometry.PointCloud, resolution) -> o3d.geometry.PointCloud:
    """
    Downsample PCD.
    """
    points = np.array(pcd.points)
    bbox = points.max(0) - points.min(0)
    d_volume = bbox.max() / resolution
    return pcd.voxel_down_sample(d_volume)


def color_pcd(pcd: o3d.geometry.PointCloud, color) -> o3d.geometry.PointCloud:
    """
    Downsample PCD.
    """
    pcd.colors = o3d.utility.Vector3dVector(
        np.array(color, float).flatten()[None, :].repeat(len(pcd.points), 0))
    return pcd
