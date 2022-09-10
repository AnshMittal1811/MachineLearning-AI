import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import config.config_loader as cfg_loader
import data_processing.utils as utils
import traceback
import tqdm


def sample_colors(gt_mesh_path):
    try:
        path = os.path.normpath(gt_mesh_path)
        challange = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]

        out_file = cfg['data_path'] + '/{}/{}/{}_color_samples{}_sigma{}_bbox{}.npz' \
            .format(split, gt_file_name, full_file_name, num_points, sigma, cfg['data_bounding_box_str'])

        if os.path.exists(out_file):
            print('File exists. Done.')
            return

        gt_mesh = utils.as_mesh(trimesh.load(gt_mesh_path))
        sample_points, face_idxs = gt_mesh.sample(
            num_points, return_index=True)

        normals = gt_mesh.face_normals[face_idxs]
        triangles = gt_mesh.triangles[face_idxs]
        face_vertices = gt_mesh.faces[face_idxs]
        faces_uvs = gt_mesh.visual.uv[face_vertices]

        q = triangles[:, 0]
        u = triangles[:, 1]
        v = triangles[:, 2]

        uvs = []

        for i, p in enumerate(sample_points):
            barycentric_weights = utils.barycentric_coordinates(
                p, q[i], u[i], v[i])
            uv = np.average(faces_uvs[i], 0, barycentric_weights)
            uvs.append(uv)

        sample_points = sample_points + sigma * \
            np.random.randn(num_points, 1) * normals

        texture = gt_mesh.visual.material.image

        colors = trimesh.visual.color.uv_to_color(np.array(uvs), texture)

        np.savez(out_file, points=sample_points, grid_coords=utils.to_grid_sample_coords(
            sample_points, bbox), colors=colors[:, :3])
    except Exception as err:
        print('Error with {}: {}'.format(out_file, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run color sampling. Samples surface points on the GT objects, and saves their coordinates along with the RGB color at their location.'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    # parser.add_argument('--sigma', type=float, default=0.015)
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    num_points = cfg['preprocessing']['color_sampling']['sample_number']
    bbox = cfg['data_bounding_box']
    sigma = cfg['preprocessing']['color_sampling']['sigma']

    print('Fining all gt object paths for point and RGB sampling.')
    paths = glob(cfg['data_path'] + cfg['preprocessing']
                 ['color_sampling']['input_files_regex'])

    print('Start sampling.')
    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(sample_colors, paths), total=len(paths)):
        pass
    p.close()
    p.join()
