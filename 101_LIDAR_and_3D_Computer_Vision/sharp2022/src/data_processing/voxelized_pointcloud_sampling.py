import data_processing.utils as utils
import tqdm
import traceback
import config.config_loader as cfg_loader
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse

# ROOT = 'shapenet/data/'

# def voxelized_pointcloud_sampling(partial_mesh_path):
#     try:
#         path = os.path.normpath(partial_mesh_path)
#         gt_file_name = path.split(os.sep)[-2]
#         full_file_name = path.split(os.sep)[-1][:-4]

#         out_file = os.path.dirname(partial_mesh_path) + '/{}_voxelized_point_cloud_res{}_points{}_bbox{}.npz'\
#             .format(full_file_name, res, num_points, bbox_str)

#         if os.path.exists(out_file):
#             print('File exists. Done.')
#             return

#         # color from partial input
#         partial_mesh = utils.as_mesh(trimesh.load(partial_mesh_path))

#         colored_point_cloud, face_idxs = partial_mesh.sample(num_points, return_index = True)

#         triangles = partial_mesh.triangles[face_idxs]
#         face_vertices = partial_mesh.faces[face_idxs]
#         faces_uvs = partial_mesh.visual.uv[face_vertices]

#         q = triangles[:, 0]
#         u = triangles[:, 1]
#         v = triangles[:, 2]

#         uvs = []

#         for i, p in enumerate(colored_point_cloud):
#             barycentric_weights = utils.barycentric_coordinates(p, q[i], u[i], v[i])
#             uv = np.average(faces_uvs[i], 0, barycentric_weights)
#             uvs.append(uv)

#         partial_texture = partial_mesh.visual.material.image

#         colors = trimesh.visual.color.uv_to_color(np.array(uvs), partial_texture)

#         occupancies = np.zeros(len(grid_points), dtype=np.int8)

#         _, idx = kdtree.query(colored_point_cloud)
#         occupancies[idx] = 1

#         compressed_occupancies = np.packbits(occupancies)


#         np.savez(out_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies, bb_min = bb_min, bb_max = bb_max, res = args.res)
#         print('Finished {}'.format(path))

#     except Exception as err:
#         print('Error with {}: {}'.format(path, traceback.format_exc()))


def voxelized_pointcloud_sampling(path):
    try:
        path = os.path.normpath(path)
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]
        split_name = path.split(os.sep)[-3]

        if not args.smpl:
            out_file = os.path.dirname(path) + '/{}_voxelized_point_cloud_res{}_points{}_bbox{}.npz'\
                .format(full_file_name, res, num_points, bbox_str)
        else:
            if split_name == "train_partial":
                split_name = "train_smpl"
            elif split_name == "test_partial":
                split_name = "test_smpl"

            smpl_file_name = f"{gt_file_name}_pose_smpl_model.obj"
            # smpl_path = os.path.join(cfg["data_path"], split_name, gt_file_name, smpl_file_name)
            # smpl_path = os.path.join(cfg["data_path"], 'pose_estimation_0504', gt_file_name, smpl_file_name)
            smpl_path = os.path.join(
<<<<<<< HEAD
                '/scratch-second/3dv/SHARP2022/pose_estimation_test_partial', gt_file_name, full_file_name + '_pose_smpl_model.obj')
=======
                '/itet-stor/leilil/net_scratch/dataset/SHARP2022/pose_estimation_0504', gt_file_name, full_file_name + '_pose_smpl_model.obj')
>>>>>>> geometry
            out_file = os.path.dirname(path) + '/{}_voxelized_point_cloud_res{}_points{}_bbox{}_smpl_estimated.npz'\
                .format(full_file_name, res, num_points, bbox_str)

        if os.path.exists(out_file):
            print('File exists. Done.')
            return

        mesh = utils.as_mesh(trimesh.load(path))
        point_cloud = mesh.sample(num_points)

        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)

        if args.smpl:
            smpl_mesh = utils.as_mesh(trimesh.load(smpl_path))
            smpl_point_cloud = mesh.sample(num_points)
            smpl_occupancies = np.zeros(len(grid_points), dtype=np.int8)
            _, idx = kdtree.query(smpl_point_cloud)
            occupancies[idx] = 1
            smpl_compressed_occupancies = np.packbits(smpl_occupancies)
            np.savez(out_file, point_cloud=point_cloud, compressed_occupancies=compressed_occupancies, res=res,
                     smpl_point_cloud=smpl_point_cloud, smpl_compressed_occupancies=smpl_compressed_occupancies)
        else:
            np.savez(out_file, point_cloud=point_cloud,
                     compressed_occupancies=compressed_occupancies, res=res)
        print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates the input for the network: a partial colored shape and a uncolorized, but completed shape. \
        Both encoded as 3D voxel grids for usage with a 3D CNN.'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--smpl', action="store_true")
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    # shorthands
    bbox = cfg['data_bounding_box']
    res = cfg['input_resolution']
    num_points = cfg['input_points_number']
    bbox_str = cfg['data_bounding_box_str']

    grid_points = utils.create_grid_points_from_xyz_bounds(*bbox, res)
    kdtree = KDTree(grid_points)

    print('Fining all input partial paths for voxelization.')
    paths = glob(cfg['data_path'] + cfg['preprocessing']
                 ['voxelized_pointcloud_sampling']['input_files_regex'])

    print('Start voxelization.')
    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(voxelized_pointcloud_sampling, paths), total=len(paths)):
        pass
    p.close()
    p.join()
