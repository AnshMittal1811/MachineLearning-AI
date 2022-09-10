import trimesh
import numpy as np
import data_processing.implicit_waterproofing as iw
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback
import data_processing.utils as utils
import tqdm
import config.config_loader as cfg_loader
import mesh_to_sdf

def scale_to_unit_cube(mesh):
    vertices = mesh.vertices
    mesh_min = bbox[::2]
    mesh_max = bbox[1::2]
    vertices = 2 * 1 / (mesh_max - mesh_min) * (vertices - mesh_min) - 1
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_to_unit_cube_points(points):
    min_values, max_values = bbox[::2], bbox[1::2]
    points = 2 * 1 / (max_values - min_values) * (points - min_values) - 1
    return points


def boundary_delta_sdf_sampling(mesh_path):
    try:

        # if os.path.exists(path +'/boundary_{}_samples.npz'.format(args.sigma)):
        #     return

        # off_path = path + '/isosurf_scaled.off'
        # out_file = path +'/boundary_{}_samples.npz'.format(args.sigma)

        path = os.path.normpath(mesh_path)
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]

        out_file = os.path.dirname(mesh_path) + '/{}_boundary_{}_sdf_samples.npz'\
            .format(full_file_name, args.sigma)
        
        smpl_mesh_path = ""
        
        # print(out_file)
        if os.path.exists(out_file):
            return

        mesh = utils.as_mesh(trimesh.load(mesh_path))
        points, face_idxs = mesh.sample(num_points, return_index = True)


        # mesh = trimesh.load(off_path)
        # points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(num_points, 3)

        scaled_mesh = scale_to_unit_cube(mesh)
        scaled_points = scale_to_unit_cube_points(boundary_points)
        sdf = mesh_to_sdf.mesh_to_sdf(scaled_mesh, scaled_points, surface_point_method='sample')


        np.savez(out_file, points=boundary_points, sdf = sdf, grid_coords= utils.to_grid_sample_coords(boundary_points, bbox))
        print('Finished {}'.format(out_file))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--sigma', type=float)
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    num_points = cfg['preprocessing']['geometry_sampling']['sample_number']
    bbox = cfg['data_bounding_box']

    # boundary_sdf_sampling("dataset/SHARP2022/train/170410-011-a-ftrm-34b3-low-res-result/170410-011-a-ftrm-34b3-low-res-result_normalized.obj")
    
    print('Fining all gt object paths for point sampling.')
    paths = glob(cfg['data_path'] + cfg['preprocessing']['geometry_sampling']['input_files_regex'])
    
    print('Start sampling.')    
    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(boundary_sdf_sampling, paths), total=len(paths)):
        pass
    p.close()
    p.join()
