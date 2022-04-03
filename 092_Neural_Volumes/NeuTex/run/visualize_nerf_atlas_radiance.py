import os
import copy
import torch
import numpy as np
import time
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from utils.visualizer import Visualizer
from utils import format as fmt
from PIL import Image
from matplotlib.colors import cnames, hex2color
import open3d
from utils.cube_map import merge_cube_to_single_texture
from models.diff_render_func import simple_tone_map


def main():
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()
    opt.is_train = False

    assert opt.resume_dir is not None

    resume_dir = opt.resume_dir
    states = torch.load(
        os.path.join(resume_dir, "{}_states.pth".format(opt.resume_epoch))
    )
    epoch_count = states["epoch_count"]
    total_steps = states["total_steps"]
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Resume from {} epoch".format(opt.resume_epoch))
    print("Iter: ", total_steps)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # data_loader = create_data_loader(opt)
    dataset = create_dataset(opt)
    pos = dataset.center_cam_pos
    viewdir = -pos / np.linalg.norm(pos)

    # load model
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    rootdir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(rootdir, exist_ok=True)

    points, normals = model.visualize_atlas()
    points = points.data.cpu().numpy()
    normals = normals.data.cpu().numpy()
    colors = np.array([hex2color(v) for v in cnames.values()])

    pcd_colors = []
    for p, c in zip(points, colors):
        pcd_colors.append(c + np.zeros_like(p))
    pcd_points = np.concatenate(points, axis=0)
    pcd_colors = np.concatenate(pcd_colors, axis=0)
    pcd_normals = np.concatenate(normals, axis=0)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcd_points)
    pcd.normals = open3d.utility.Vector3dVector(pcd_normals)
    pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    # open3d.visualization.draw_geometries([pcd])
    open3d.io.write_point_cloud(os.path.join(rootdir, "visualize.pcd"), pcd)

    meshes, textures = model.visualize_mesh_3d(icosphere_division=7)
    for i, (mesh, texture) in enumerate(zip(meshes, textures)):
        color = (255 * texture.data.cpu().numpy().clip(0, 1)).astype(np.uint8)
        c = np.ones((len(color), 4)) * 255
        c[:, :3] = color
        import trimesh

        mesh.visual.vertex_colors = np.ones_like(c)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)
        mesh.show(viewer="gl", smooth=True)

        mesh.visual.vertex_colors = c
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)
        mesh.show(viewer="gl", smooth=True)
        mesh.export(os.path.join(rootdir, "mesh_{}.ply".format(i)))

    net_texture = model.net_nerf_atlas.module.net_texture
    assert net_texture.__class__.__name__ == "TextureViewMlpMix"

    if opt.primitive_type == "sphere":
        from tqdm import tqdm

        imgs = []
        imgs2 = []
        for i, pos in enumerate(tqdm(dataset.campos)):
            viewdir = -pos / np.linalg.norm(pos)
            texture = net_texture.textures[0].export_textures(512, viewdir) ** (1 / 2.2)
            texture = merge_cube_to_single_texture(texture)
            texture = texture.clamp(0, 1).data.cpu().numpy()
            imgs.append(texture)

            Image.fromarray((texture * 255).astype(np.uint8)).save(
                os.path.join(rootdir, f"cube_view_{i}.png")
            )

            texture = net_texture.textures[0]._export_sphere(512, viewdir) ** (1 / 2.2)
            texture = texture.clamp(0, 1).data.cpu().numpy()
            imgs2.append(texture)

            Image.fromarray((texture * 255).astype(np.uint8)).save(
                os.path.join(rootdir, f"sphere_view_{i}.png")
            )

        # texture = np.max(np.array(imgs), axis=0)
        # Image.fromarray((texture * 255).astype(np.uint8)).save(
        #     os.path.join(rootdir, "cube_view.png")
        # )

    else:
        texture = net_texture.textures[0].export_textures(512, None)
        texture = texture.clamp(0, 1).data.cpu().numpy()
        Image.fromarray((texture * 255).astype(np.uint8)).save(
            os.path.join(rootdir, "square.png")
        )

        texture = net_texture.textures[0].export_textures(512, viewdir) ** (1 / 2.2)
        texture = texture.clamp(0, 1).data.cpu().numpy()
        Image.fromarray((texture * 255).astype(np.uint8)).save(
            os.path.join(rootdir, "square_view.png")
        )


if __name__ == "__main__":
    main()
