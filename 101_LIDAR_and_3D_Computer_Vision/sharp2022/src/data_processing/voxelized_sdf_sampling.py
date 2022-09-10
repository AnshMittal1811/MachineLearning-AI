from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import config.config_loader as cfg_loader
import traceback
import tqdm
import data_processing.utils as utils
import os
import mcubes
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import mesh_to_sdf

ROOT = 'shapenet/data/'

def scale_to_unit_cube(mesh):
    mesh_min = bbox[::2]
    mesh_max = bbox[1::2]
    vertices = mesh.vertices
    vertices = 2 * 1 / (mesh_max - mesh_min) * (vertices - mesh_min) - 1
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def mesh_to_voxels(mesh, voxel_resolution=64, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, pad=False, check_result=False, return_gradients=False):
    mesh = scale_to_unit_cube(mesh)
    surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method, 3**0.5, scan_count, scan_resolution, sample_point_count, sign_method=='normal')
    return surface_point_cloud.get_voxels(voxel_resolution, sign_method=='depth', normal_sample_count, pad, check_result, return_gradients)

def voxelized_sdf_sampling(path):
    try:
        path = os.path.normpath(path)
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]
        split_name = path.split(os.sep)[-3]

        if not args.smpl:
            out_file = os.path.dirname(path) + '/{}_voxelized_sdf_res{}_points{}_bbox{}.npz'\
                .format(full_file_name, res, num_points, bbox_str)
        else: 
            if split_name == "train_partial":
                split_name = "train_smpl"
            elif  split_name == "test_partial":
                split_name = "test_smpl"

            smpl_file_name = f"{gt_file_name}_pose_smpl_model.obj"
            smpl_path = os.path.join(cfg["data_path"], split_name, gt_file_name, smpl_file_name)
            out_file = os.path.dirname(path) + '/{}_voxelized_sdf_res{}_points{}_bbox{}_smpl.npz'\
                .format(full_file_name, res, num_points, bbox_str)

        if os.path.exists(out_file):
            print('File exists. Done.')
            return


        mesh = utils.as_mesh(trimesh.load(path))
        sdf = mesh_to_voxels(mesh, voxel_resolution=128, surface_point_method='sample')
        vertices, triangles = mcubes.marching_cubes(sdf, 0)
        mesh =  trimesh.Trimesh(vertices, triangles)
        mesh.export('test_sdf.obj')

        np.savez(out_file, sdf = sdf, res = res)
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
    paths = glob(cfg['data_path'] + cfg['preprocessing']['voxelized_pointcloud_sampling']['input_files_regex'])

    #debug
    voxelized_sdf_sampling("dataset/SHARP2022/train/170410-011-a-ftrm-34b3-low-res-result/170410-011-a-ftrm-34b3-low-res-result_normalized.obj")
    
    print('Start voxelization.')
    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(voxelized_sdf_sampling, paths), total=len(paths)):
        pass
    p.close()
    p.join()