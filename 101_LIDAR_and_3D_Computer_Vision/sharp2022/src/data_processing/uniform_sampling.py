
import os
import sys
import config.config_loader as cfg_loader
import tqdm
import data_processing.utils as utils
import traceback
import argparse
from multiprocessing import Pool
import multiprocessing as mp
from glob import glob
import data_processing.implicit_waterproofing as iw
import numpy as np
import trimesh


def boundary_sampling(mesh_path):
    try:

        # if os.path.exists(path +'/boundary_{}_samples.npz'.format(args.sigma)):
        #     return

        # off_path = path + '/isosurf_scaled.off'
        # out_file = path +'/boundary_{}_samples.npz'.format(args.sigma)

        path = os.path.normpath(mesh_path)
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]

        out_file = os.path.dirname(mesh_path) + '/{}_boundary_uniform_samples.npz'\
            .format(full_file_name)

        # print(out_file)
        if os.path.exists(out_file):
            return
        grid_points = utils.create_grid_points_from_xyz_bounds(
            *bbox, resolution)
        # grid_coords = utils.to_grid_sample_coords(grid_points, bbox)
        grid_points = grid_points.reshape([len(grid_points), 3])

        subsample_indices = np.random.randint(0, len(grid_points), num_points)

        grid_points = grid_points[subsample_indices]

        mesh = utils.as_mesh(trimesh.load(mesh_path))
        points, face_idxs = mesh.sample(num_points, return_index=True)

        # mesh = trimesh.load(off_path)
        # points = mesh.sample(sample_num)

        occupancies = iw.implicit_waterproofing(mesh, grid_points)[0]

        np.savez(out_file, points=grid_points, occupancies=occupancies,
                 grid_coords=utils.to_grid_sample_coords(grid_points, bbox))
        print('Finished {}'.format(out_file))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    # parser.add_argument('--sigma', type=float)
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    num_points = cfg['preprocessing']['geometry_sampling']['sample_number']
    bbox = cfg['data_bounding_box']
    resolution = cfg['input_resolution']

    print('Fining all gt object paths for point sampling.')
    paths = glob(cfg['data_path'] + cfg['preprocessing']
                 ['geometry_sampling']['input_files_regex'])

    print('Start sampling.')
    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(boundary_sampling, paths), total=len(paths)):
        pass
    p.close()
    p.join()
