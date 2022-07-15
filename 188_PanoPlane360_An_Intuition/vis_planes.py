import open3d
import argparse
import numpy as np
from imageio import imread
from utils import read_h_planes, read_v_planes
import torch

from models import models_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', required=True)
    parser.add_argument('--h_planes', required=True)
    parser.add_argument('--v_planes', required=True)
    parser.add_argument('--mesh', action='store_true')
    parser.add_argument('--mesh_show_back_face', action='store_true')
    args = parser.parse_args()

    # Read input
    v_planes = imread(args.v_planes)
    h_planes = imread(args.h_planes)
    H_, W = h_planes.shape
    cropped = (W//2 - H_) // 2
    h_planes[H_//2-10:H_//2+10] = 0
    h_planar = h_planes != 0
    v_planar = np.abs(v_planes).sum(-1) != 0
    rgb = imread(args.img)[..., :3]
    if cropped > 0:
        rgb = rgb[cropped:-cropped]

    # Planes to depth
    v2d = models_utils.vplane_2_depth(torch.FloatTensor(v_planes.transpose(2, 0, 1)[None]))[0, 0]
    h2d = models_utils.hplane_2_depth(torch.FloatTensor(h_planes[None, None]))[0, 0]
    depth = np.zeros([H_, W])
    depth[v_planar] = v2d[v_planar]
    depth[h_planar] = h2d[h_planar]
    depth = np.clip(depth, 0, 20)

    # Project to 3d
    v_grid = models_utils.v_grid(1, 1, H_, W)[0, 0].numpy()  # H_, W
    u_grid = models_utils.u_grid(1, 1, H_, W)[0, 0].numpy()  # H_, W
    zs = depth * np.sin(v_grid)
    xs = depth * np.cos(v_grid) * np.cos(u_grid)
    ys = depth * np.cos(v_grid) * np.sin(u_grid)
    pts_xyz = np.stack([xs, ys, zs], -1).reshape(-1, 3)
    pts_rgb = rgb.reshape(-1, 3) / 255

    if args.mesh:
        pid = np.arange(len(pts_xyz)).reshape(H_, W)
        tri_cancididate = np.concatenate([
            np.stack([
                pid[:-1, :-1], pid[1:, :-1], np.roll(pid, -1, axis=1)[:-1, :-1],
            ], -1),
            np.stack([
                pid[1:, :-1], np.roll(pid, -1, axis=1)[1:, :-1], np.roll(pid, -1, axis=1)[:-1, :-1],
            ], -1)
        ])
        vparams, vid_mask = np.unique(v_planes.reshape(-1, 3), return_inverse=True, axis=0)
        hparams, hid_mask = np.unique(h_planes.reshape(-1), return_inverse=True, axis=0)
        faces = []
        for i in range(len(vparams)):
            if np.abs(vparams[i]).sum() == 0:
                continue
            mask = (vid_mask == i)
            masktri = (mask[tri_cancididate].sum(-1) == 3)
            faces.extend(tri_cancididate[masktri])
        for i in range(len(hparams)):
            if hparams[i] == 0:
                continue
            mask = (hid_mask == i)
            masktri = (mask[tri_cancididate].sum(-1) == 3)
            faces.extend(tri_cancididate[masktri])
        scene = open3d.geometry.TriangleMesh()
        scene.vertices = open3d.utility.Vector3dVector(pts_xyz)
        scene.vertex_colors = open3d.utility.Vector3dVector(pts_rgb)
        scene.triangles = open3d.utility.Vector3iVector(faces)
    else:
        scene = open3d.geometry.PointCloud()
        scene.points = open3d.utility.Vector3dVector(pts_xyz)
        scene.colors = open3d.utility.Vector3dVector(pts_rgb)

    open3d.visualization.draw_geometries([
        scene,
        open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    ], mesh_show_back_face=args.mesh_show_back_face)

