# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import logging
import functools
import os
import os.path as osp
import pickle
import time
from typing import List, Tuple, Union, Callable
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.shader import HardPhongShader,  SoftPhongShader
from pytorch3d.renderer.mesh.shading import flat_shading
import skimage.measure

import numpy as np
from scipy.spatial.distance import cdist
import re
from torch._six import string_classes
import collections.abc as container_abcs

import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.io as io3d
import pytorch3d.ops as ops_3d

import pytorch3d.structures
import pytorch3d.structures.utils as struct_utils
from pytorch3d.structures import Pointclouds
from pytorch3d.structures.meshes import join_meshes_as_batch
from pytorch3d.transforms import Transform3d, euler_angles_to_matrix, Rotate, Scale, Translate
from pytorch3d.renderer import (
    MeshRasterizer, PointsRasterizer,
    PointsRenderer,
    DirectionalLights,
    TexturesUV, TexturesVertex,
    AlphaCompositor, 
    RasterizationSettings, PointsRasterizationSettings, PerspectiveCameras, BlendParams,
    )
from trimesh.base import Trimesh
from trimesh.voxel.base import VoxelGrid
from . import image_utils, geom_utils
from .layers import grid_sample

from .my_pytorch3d import Meshes, chamfer_distance
# ### Mesh IO Utils ###
def meshfile_to_glb(mesh_file, out_file):
    mesh = trimesh.load(mesh_file)  
    if not out_file.endswith('.glb'):
        out_file += '.glb'
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    mesh.export(file_type="glb", file_obj=out_file)  
    print('export to ', mesh)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    return mesh


def load_mesh_from_np(verts, faces, device='cpu'):
    """
    :param verts: (V, 3)
    :param faces: (F, 3)
    :param device:
    :return:
    """
    meshes = Meshes([torch.FloatTensor(verts)], [torch.FloatTensor(faces)]).to(device)
    return meshes


def load_mesh(mesh_path, device='cpu', scale_verts=1, scale_rgb=255, texture='vertex'):
    """
    :param mesh_path:
    :param device: str
    :param scale_verts: default=1
    :param scale_rgb:
    :param texture:
    :return: Meshes: texture scale: 0,1
    """
    mesh = trimesh.load(mesh_path, process=False)
    # print('trimesh load', mesh.faces[0:3])
    mesh = as_mesh(mesh)

    verts = torch.FloatTensor(mesh.vertices).to(device) * scale_verts
    faces = torch.LongTensor(mesh.faces).to(device)

    if texture == 'uv':
        uv = torch.FloatTensor(mesh.visual.uv, ).to(device)
        color_map = np.asanyarray(mesh.visual.material.image.convert('RGBA'))[:, :, :3] / scale_rgb
        color_map = torch.FloatTensor(color_map, ).to(device)
        texture = TexturesUV(maps=[color_map], verts_uvs=[uv], faces_uvs=[faces])
    elif texture == 'vertex':
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            visual = mesh.visual.to_color()
        else:
            visual = mesh.visual
        try:
            verts_rgb = visual.vertex_colors
            verts_rgb = verts_rgb[:, :3] / scale_rgb
        except:
            verts_rgb = np.ones_like(verts)
        verts_rgb = torch.FloatTensor(verts_rgb).to(device)
        texture = TexturesVertex([verts_rgb])
    elif texture == 'white':
        texture = TexturesVertex([torch.ones_like(verts)])

    mesh = Meshes([verts], [faces], texture)
    return mesh


def dump_voxes(filepath, voxels):
    """
    Args:
        filepath ([type]): [description]
        voxels ([type]): (N, 1, D, H, W)
    """
    N = len(voxels)
    if isinstance(filepath, str):
        path_prefix = filepath
        filepath = [path_prefix + '_%d' % n for n in range(N)]
    elif isinstance(filepath, list):
        assert len(filepath) == len(meshes)
    voxels = voxels.cpu().detach().numpy()
    print('save voxels to ', filepath[0])

    # def save_one(filepath, voxes)    :
    #     os.makedirs(os.path.dirname(filepath), exist_ok=True)
    #     np.savez_compressed(filepath + '.npz', vox=voxes)
    
    # Parallel(8, verbose=5)(
    #     delayed(save_one)(filepath[n], voxels[n, 0], )
    #     for n in range(N)
    # )
    for n in range(N):
        os.makedirs(os.path.dirname(filepath[n]), exist_ok=True)
        np.savez_compressed(filepath[n] + '.npz', vox=voxels[n, 0])


def dump_meshes(mesh_path, meshes: Meshes, ext='.obj'):
    """
    :param mesh_path: str or list of str
    :param meshes:
    :return:
    """
    N = len(meshes)
    if isinstance(mesh_path, str):
        mesh_prefix = mesh_path
        mesh_path = [mesh_prefix + '_%d' % n for n in range(N)]
    elif isinstance(mesh_path, list):
        assert len(mesh_path) == len(meshes)

    meshes = meshes.to('cpu')

    verts = meshes.verts_list()
    faces = meshes.faces_list()
    if len(verts) == 0:
        print('skip empty meshes')
        return

    print('save meshes to ', mesh_path[0])

    for n in range(N):
        os.makedirs(os.path.dirname(mesh_path[n]), exist_ok=True)
        if ext == '.obj':
            io3d.save_obj(mesh_path[n] + ext, verts[n], faces[n])
        else:
            io3d.save_ply(mesh_path[n] + ext, verts[n], faces[n])

def trans_coord(meshes: Meshes, rad=-np.pi / 2, axis='X'):
    """
    Transofrm coordinate from y up to z up, etc
    :return:
    """
    device = meshes.device
    N = len(meshes)
    angles = torch.zeros([N, 3], device=device)
    angles[:, 0] += rad
    trans = Rotate(euler_angles_to_matrix(angles, axis + 'YX'), device=device, )
    meshes = meshes.update_padded(trans.transform_points(meshes.verts_padded()))
    return meshes


def flip_faces(meshes: Meshes):
    faces_list = meshes.faces_list()
    faces_list = [torch.stack([faces[..., 1], faces[..., 0], faces[..., 2]], dim=-1)
                  for faces in faces_list]
    new_meshes = Meshes(meshes.verts_list(), faces_list, meshes.textures)
    return new_meshes


def load_mesh_from_np(vertices, faces, device='cpu', scale_rgb=255, texture=None) -> Meshes:
    verts = torch.FloatTensor(vertices, ).to(device)
    faces = torch.LongTensor(faces.astype(np.int64)).to(device)

    if texture is None:
        verts_rgb = np.ones_like(verts)
    else:
        verts_rgb = texture[:, :3] / scale_rgb

    verts_rgb = torch.FloatTensor(verts_rgb).to(device)
    texture = TexturesVertex([verts_rgb])

    mesh = Meshes([verts], [faces], texture)
    return mesh


def pad_texture(meshes: Meshes, feature: torch.Tensor='white') -> TexturesVertex:
    """
    :param meshes:
    :param feature: (sumV, C)
    :return:
    """
    if isinstance(feature, TexturesVertex):
        return feature
    if feature == 'white':
        feature = torch.ones_like(meshes.verts_padded())
    elif feature == 'blue':
        feature = torch.zeros_like(meshes.verts_padded())
        color = torch.FloatTensor([[[203,238,254]]]).to(meshes.device)  / 255   
        color = torch.FloatTensor([[[183,216,254]]]).to(meshes.device)  / 255   
        feature = feature + color
    elif feature == 'yellow':
        feature = torch.zeros_like(meshes.verts_padded())
        # yellow = [250 / 255.0, 230 / 255.0, 154 / 255.0],        
        color = torch.FloatTensor([[[250 / 255.0, 230 / 255.0, 154 / 255.0]]]).to(meshes.device) * 2 - 1
        feature = feature + color
    elif feature == 'random':
        feature = torch.rand_like(meshes.verts_padded())  # [0, 1]
    if feature.dim() == 2:
        feature = struct_utils.packed_to_list(feature, meshes.num_verts_per_mesh().tolist())
        # feature = struct_utils.list_to_padded(feature, pad_value=-1)

    texture = TexturesVertex(feature)
    texture._num_faces_per_mesh = meshes.num_faces_per_mesh().tolist()
    texture._num_verts_per_mesh = meshes.num_verts_per_mesh().tolist()
    texture._N = meshes._N
    texture.valid = meshes.valid
    return texture


# ### Gripper Utils ###
def gripper_mesh(se3=None, mat=None, texture=None, return_mesh=True):
    """
    :param se3: (N, 6)
    :param mat:
    :return:
    """
    if mat is None:
        mat = geom_utils.se3_to_matrix(se3)  # N, 4, 4
    device = mat.device

    verts, faces = create_gripper(mat.device)  # (1, V, 3), (1, V, 3)
    t = Transform3d(matrix=mat.transpose(1, 2), device=device)

    verts = t.transform_points(verts)
    faces = faces.expand(mat.size(0), faces.size(1), 3)

    if texture is not None:
        texture = texture.unsqueeze(1).to(device) + torch.zeros_like(verts)
    else:
        texture = torch.ones_like(verts)

    if return_mesh:
        return Meshes(verts, faces, TexturesVertex(texture))
    else:
        return verts, faces, texture


# ### Primitives Utils ###
def create_cube(device, N=1, align='center'):
    """
    :return: verts: (1, 8, 3) faces: (1, 12, 3)
    """
    cube_verts = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    if align == 'center':
        cube_verts -= .5

    # faces corresponding to a unit cube: 12x3
    cube_faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ],
        dtype=torch.int64,
        device=device,
    )  # 12, 3

    return cube_verts.unsqueeze(0).expand(N, 8, 3), cube_faces.unsqueeze(0).expand(N, 12, 3)


def create_gripper(device, N=1):
    """
    :param: texture: (N, 3) in scale [-1, 1] or None
    :return: torch.Tensor in shape of (N, V, 3), (N, V, 3)"""
    scale = Scale(torch.tensor(
        [
            [0.005, 0.005, 0.139],
            [0.005, 0.005, 0.07],
            [0.005, 0.005, 0.06],
            [0.005, 0.005, 0.06],
        ], dtype=torch.float32, device=device
    ), device=device)
    translate = Translate(torch.tensor(
        [
            [-0.03, 0, 0, ],
            [-0.065, 0, 0, ],
            [0, 0, 0.065, ],
            [0, 0, -0.065, ],
        ], dtype=torch.float32, device=device
    ), device=device)
    rot = Rotate(euler_angles_to_matrix(torch.tensor(
        [
            [0, 0, 0],
            [0, np.pi / 2, 0],
            [0, np.pi / 2, 0],
            [0, np.pi / 2, 0],
        ], dtype=torch.float32, device=device
    ), 'XYZ'), device=device)
    align = Rotate(euler_angles_to_matrix(torch.tensor(
        [
            [np.pi / 2, 0, 0],
        ], dtype=torch.float32, device=device
    ), 'XYZ'), device=device)
    # X -> scale -> R -> t, align
    transform = scale.compose(rot, translate, align)

    each_verts, each_faces, num_cube = 8, 12, 4
    verts, faces = create_cube(device, num_cube)
    verts = (transform.transform_points(verts)).view(1, num_cube * each_verts, 3)  # (4, 8, 3) -> (1, 32, 3)
    offset = torch.arange(0, num_cube, device=device).unsqueeze(-1).unsqueeze(-1) * each_verts  # faces offset
    faces = (faces + offset).view(1, num_cube * each_faces, 3)

    verts = verts.expand(N, num_cube * each_verts, 3)
    faces = faces.expand(N, num_cube * each_faces, 3)

    return verts, faces


# ### Pointcloud to meshes Utils ###
def pc_to_cubic_meshes(xyz: torch.Tensor = None, feature: torch.Tensor = None, pc: Pointclouds = None,
                       align='center', eps=None) -> Meshes:
    if pc is None:
        if feature is None:
            feature = torch.ones_like(xyz)
        pc = Pointclouds(xyz, features=feature)
    device = pc.device
    if pc.isempty():
        N = len(pc)
        zeros = torch.zeros([N, 0, 3], device=device)
        meshes = Meshes(zeros, zeros, textures=TexturesVertex(zeros))
    else:
        N, V, D = pc.features_padded().size()

        norm = torch.sqrt(torch.sum(pc.points_padded() ** 2, dim=-1, keepdim=True))
        std = torch.std(norm, dim=1, keepdim=True)  # (N, V, 3)
        if eps is None:
            eps = (std / 10).clamp(min=5e-3)  # N, 1, 1
        # eps = .1

        cube_verts, cube_faces = create_cube(device, align=align)

        num = pc.num_points_per_cloud()  # (N, )

        faces_list = [cube_faces.expand(num_e, 12, 3) for num_e in num]
        faces_offset = [torch.arange(0, num_e, device=device).unsqueeze(-1).unsqueeze(-1) * 8 for num_e in num]
        faces_list = [(each_f + each_off).view(-1, 3) for each_f, each_off in zip(faces_list, faces_offset)]

        verts = cube_verts.expand(N, 8, 3) + torch.randn([N, 8, 3], device=device) * 0.01  # (1, 8, 3)
        verts = pc.points_padded().unsqueeze(-2) + (verts * eps).unsqueeze(1)  # N, 8, 3  -> N, V, 8, 3
        verts = verts.view(N, V * 8, 3)

        num8_list = (num * 8).tolist()
        verts_list = struct_utils.padded_to_list(verts, num8_list)

        feature = pc.features_padded().unsqueeze(-2).expand(N, V, 8, D).reshape(N, V * 8, D)
        feature_list = struct_utils.padded_to_list(feature, num8_list)
        texture = TexturesVertex(feature_list)

        meshes = Meshes(verts_list, faces_list, texture)

    return meshes


# ### Render Utils ###

def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling ** 2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def sample_images_at_mc_locs(target_images, sampled_rays_xy):
    """
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this method samples the tensor `target_images` at the
    respective 2D locations.
    This function is used in order to extract the colors from
    ground truth images that correspond to the colors
    rendered using `MonteCarloRaysampler`.
    :param target_images:   (N, C, H, W)
    :param sampled_rays_xy: (N, ..., 2)
    :returns (N, ..., C)
    """
    ba = target_images.shape[0]
    dim = target_images.shape[1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    # In order to sample target_images, we utilize
    # the grid_sample function which implements a
    # bilinear image sampler.
    # Note that we have to invert the sign of the
    # sampled ray positions to convert the NDC xy locations
    # of the MonteCarloRaysampler to the coordinate
    # convention of grid_sample.
    # todo: judy - sample ray xy?
    images_sampled = grid_sample(target_images, sampled_rays_xy.view(ba, -1, 1, 2),)
    # images_sampled = torch.nn.functional.grid_sample(
    #     # target_images.permute(0, 3, 1, 2),
    #     target_images,
    #     sampled_rays_xy.view(ba, -1, 1, 2),  # note the sign inversion
    #     align_corners=True
    # )
    return images_sampled.permute(0, 2, 3, 1).view(
        ba, *spatial_size, dim
    )


def transform_points(cPoints, cameras:PerspectiveCameras, cTw=None):
    # K = get_k(cameras)
    ndcTc = cameras.get_projection_transform()
    ndcPoints = ndcTc.transform_points(cPoints)[..., :2]
    return ndcPoints

def render_mesh_flow(wMeshes, cam1: PerspectiveCameras, cam2: PerspectiveCameras,
        return_ndc=True, **kwargs):
    """
    Args:
        wmeshes (_type_): _description_
        c1Tw (_type_): view transform in shape of (N, 4, 4)
        c2Tw (_type_): view transform of next frame (N, 4, 4)
    Returns:
        flow1 to 2: N, H, W, 2 in pixel space
    """
    image_size = kwargs.get('out_size', 224)
    c1Tw = kwargs.get('c1Tw', None)
    c2Tw = kwargs.get('c2Tw', None)
    raster_settings = kwargs.get('raster_settings',
                                 RasterizationSettings(
                                     image_size=image_size, 
                                     faces_per_pixel=1,
                                    perspective_correct=False,
                                     cull_backfaces=False))
    device = cam1.device

    rasterizer = MeshRasterizer(cameras=cam1, raster_settings=raster_settings).to(device)
    if c1Tw is not None:
        cam1Meshes = apply_transform(wMeshes, c1Tw)
    else:
        cam1Meshes = wMeshes
    out = {}
    fragments = rasterizer(cam1Meshes, **kwargs)

    # unproject and reproject
    zbuf = fragments.zbuf[..., 0]  # (N, H, W, 1)
    N, H, W = zbuf.size()

    # flipped image
    ys, xs = torch.meshgrid(
        torch.linspace(1, -1, H),
        torch.linspace(1, -1, W)
    )  # (H, W)
    ys = ys.unsqueeze(0).expand(N, H, W).to(device)
    xs = xs.unsqueeze(0).expand(N, H, W).to(device)
    i1Xy = torch.stack([xs, ys], -1)

    cam1Xyz = unproj_depth_to_xyz(zbuf, cam1, scale=False, xy=i1Xy)
    if c1Tw is not None:
        wXyz = apply_transform(cam1Xyz, geom_utils.inverse_rt(mat=c1Tw, return_mat=True))
    else:
        wXyz = cam1Xyz

    if c2Tw is not None:
        c2Xyz = apply_transform(wXyz, c2Tw)
    else:
        c2Xyz = wXyz

    i2Xy = cam2.transform_points_ndc(c2Xyz.view(N, H*W, 3))[..., :2].view(N, H, W, 2)
    invalid_mask = zbuf == -1
    flow12 = i2Xy - i1Xy
    flow12[invalid_mask] = 0

    flow12 = torch.flip(flow12, dims=[1, 2])  # flip up-down, and left-right
    invalid_mask = torch.flip(invalid_mask.float(), dims=[1, 2]).bool()
    wXyz = torch.flip(wXyz, [1, 2])
    i2Xy = torch.flip(i2Xy, [1, 2])
    if not return_ndc:
        # flow12 += 1
        flow12 *= image_size / 2
    out['flow'] = flow12
    out['wXyz'] = wXyz  # (N, H, W, 3)
    out['invalid_mask'] = invalid_mask  # (N, H, W, K)
    return out


def render_mesh(meshes: Meshes, cameras, rgb_mode=True, depth_mode=False, **kwargs):
    """
    flip issue: https://github.com/facebookresearch/pytorch3d/issues/78
    :param meshes:
    :param out_size: H=W
    :param cameras:
    :param kwargs:
    :return: 'rgb': (N, 3, H, W). 'mask': (N, 1, H, W). 'rgba': (N, 3, H, W)
    """
    image_size = kwargs.get('out_size', 224)
    raster_settings = kwargs.get('raster_settings',
                                 RasterizationSettings(
                                     image_size=image_size, 
                                     faces_per_pixel=2,
                                     cull_backfaces=False))
    device = cameras.device

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    out = {}
    fragments = rasterizer(meshes, **kwargs)
    out['frag'] = fragments

    if rgb_mode:
        # shader = HardGouraudShader(device=meshes.device, lights=ambient_light(meshes.device, cameras))
        shader = HardPhongShader(device=meshes.device, lights=ambient_light(meshes.device, cameras))
        image = shader(fragments, meshes, cameras=cameras, )  # znear=znear, zfar=zfar, **kwargs)
        rgb, _ = flip_transpose_canvas(image)

        # get mask
        # Find out how much background_color needs to be expanded to be used for masked_scatter.
        N, H, W, K = fragments.pix_to_face.shape
        is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)
        alpha = torch.ones((N, H, W, 1), dtype=rgb.dtype, device=device)
        alpha[is_background] = 0.
        mask = flip_transpose_canvas(alpha, False)

        # Concat with the alpha channel.

        out['image'] = rgb
        out['mask'] = mask
    if depth_mode:
        zbuf = fragments.zbuf[..., 0:1]
        zbuf = flip_transpose_canvas(zbuf, False)
        out['depth'] = zbuf

    return out


def render_soft(meshes: Meshes, cameras, rgb_mode=True, depth_mode=False, **kwargs):
    """
    :param meshes:
    :param cameras:
    :param kwargs:
    :return: 'image': (N, 3, H, W),
              'mask': (N, 1, H, W),
             'depth': (N, 1, H, W),
              'frag':,
    """
    blend_params = BlendParams(sigma=kwargs.get('sigma', 1e-5), gamma=1e-4)
    dist_eps = 1e-6
    # blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    # dist_eps = 1e-4
    device = cameras.device
    raster_settings = RasterizationSettings(
        image_size=kwargs.get('out_size', 224),
        blur_radius=np.log(1. / dist_eps - 1.) * blend_params.sigma,
        faces_per_pixel=kwargs.get('faces_per_pixel', 10),
        perspective_correct=False,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)

    out = {}
    fragments = rasterizer(meshes, **kwargs)
    out['frag'] = fragments

    if rgb_mode:
        # shader = SoftGouraudShader(device, lights=ambient_light(meshes.device, cameras))
        shader = SoftPhongShader(device, lights=ambient_light(meshes.device, cameras))
        if torch.isnan(fragments.zbuf).any():
            fname = '/checkpoint/yufeiy2/hoi_output/vis/mesh.pkl'
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fp:
                pickle.dump({'mesh': meshes, 'camera': cameras}, fp)
            import pdb
            pdb.set_trace()
        image = shader(fragments, meshes, cameras=cameras, )  # znear=znear, zfar=zfar, **kwargs)
        rgb, mask = flip_transpose_canvas(image)

        out['image'] = rgb
        out['mask'] = mask

    if depth_mode:
        zbuf = fragments.zbuf[..., 0:1]
        zbuf = flip_transpose_canvas(zbuf, False)
        zbuf[zbuf != zbuf] = -1
        out['depth'] = zbuf

    return out


def flip_transpose_canvas(image, rgba=True):
    image = torch.flip(image, dims=[1, 2])  # flip up-down, and left-right
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    if rgba:
        rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]
        return rgb, mask
    else:
        return image


def render_pc(pointcloud, cameras, **kwargs):
    """
    :param meshes:
    :param out_size: H=W
    :param cameras:
    :param kwargs:
    :return: 'rgb': (N, 3, H, W). 'mask': (N, 1, H, W). 'rgba': (N, 3, H, W)
    """
    device = pointcloud.device

    raster_settings = PointsRasterizationSettings(image_size=kwargs.get('out_size', 224))
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    image = renderer(pointcloud, **kwargs)

    image = torch.flip(image, dims=[-3, -2])
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    # rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]
    return {'image': image}


def render_geom(geom: Union[Meshes, Pointclouds], azel=[[0, 0]], scale_geom=False, view_centric=False, cameras=None,
                **kwargs):
    N = len(geom)
    device = geom.device
    render = get_render_func(geom)

    # 1. convert object frame
    if view_centric:
        assert cameras is not None, 'Specify camera'
        geom = transform_verts_to_view(geom, cameras)
    center_geom, xyz = center_obj(geom)

    # 2. get camera list
    if not view_centric:
        xyz = torch.zeros_like(xyz)
        xyz[:, 2] = kwargs.get('dist', 20)  # N, 3
        if scale_geom:
            geom = center_geom
            verts = get_verts(geom)  # N, V, 3 --> N, 1, 3
            verts = 2 * verts / (torch.max(verts.view(N, -1), dim=1)[0].view(N, 1, 1)
                                 - torch.min(verts.view(N, -1), dim=1)[0].view(N, 1, 1))
            geom = geom.update_padded(verts)

    if not torch.is_tensor(azel):
        azel = torch.tensor(azel, dtype=torch.float32, device=device)  # (1, 2)
    azel = azel.repeat(N, 1)

    view_cameras = param_to_cameras(azel, xyz, cameras)
    image = render(geom, view_cameras, **kwargs)
    return image


def center_norm_geom(geom,  dist=20, max_norm=1):
    """center to 0,0,0, with -1, 1? """
    device = geom.device
    N = len(geom)
    verts = get_verts(geom)  # N, V, 3 --> N, 1, 3

    bbnx_max = torch.max(verts, dim=1, keepdim=False)[0]  # (N, 3)
    bbnx_min = torch.min(verts, dim=1, keepdim=False)[0]  # (N, 3)

    width, dim = torch.max(bbnx_max - bbnx_min, dim=-1, keepdim=False)  # (N, 1, 1)
    scale = Scale(2 * max_norm / width, device=device)
    geom_norm = apply_transform(geom, scale)
    
    center = (bbnx_max + bbnx_min) /2 * 2 / width
    trans = Translate(-center, device=device)

    cGeom = apply_transform(geom_norm, trans)
    
    transfm = scale.compose(trans)

    return cGeom, transfm

def render_geom_rot(geom: Union[Meshes, Pointclouds],
                    view_mod='az', scale_geom=False, 
                    view_centric=False, cameras: PerspectiveCameras = None, **kwargs):
    """
    az / el rotate around up-axis in either object frame, or viewer frame.
    2 steps: 1) convert object frame. 2) get view list
    1.1 object frame: translate object to 0,0,0.
    1.2 camera frame: world2view transform, then translate object to 0,0,0.
    2.1 az, el: just reset for now
    :param geom: Pointclouds or Meshes
    :return:
    """
    N = len(geom)
    device = geom.device
    render = get_render_func(geom)

    if cameras is None:
        cameras = PerspectiveCameras(10, device=device)

        verts = get_verts(geom)  # N, V, 3 --> N, 1, 3
        bbnx_max = torch.max(verts, dim=1, keepdim=False)[0]  # (N, 1, 3)
        bbnx_min = torch.min(verts, dim=1, keepdim=False)[0]  # (N, 1, 3)
        width, dim = torch.max(bbnx_max - bbnx_min, dim=-1, keepdim=False)  # (N, 1, 1)

        scale = Scale(2/width, device=device)
        geom_norm = apply_transform(geom, scale)

        geom_norm, _ = center_obj(geom_norm)
        
        xyz = torch.zeros([N, 3], device=device)
        xyz[..., 2] = kwargs.get('dist', 20)
        trans = Translate(xyz, device=device)
        cGeom = apply_transform(geom_norm, trans)

    else:
        cGeom = geom
        xyz = kwargs.get('xyz', None)
        if xyz is None:
            _, xyz = center_obj(cGeom)
    # rotate around the center of canvas
    oTc = Translate(-xyz, device=device)

    oGeom = apply_transform(cGeom, oTc)
    cTo = oTc.inverse()

    # 2. get rotation list
    azel = get_view_list(view_mod, device=device, **kwargs)  # (T, 2)
    T = azel.size(0)
    azel = azel.unsqueeze(1).expand(azel.size(0), N, 2)

    image_list = []
    for t in range(T):
        rot = geom_utils.azel_to_rot(azel[t], homo=True)
        oGeom_prim = apply_transform(oGeom, rot)
        cGeom_prim = apply_transform(oGeom_prim, cTo)
        vox_recon = render(cGeom_prim, cameras, **kwargs)
        image_list.append(vox_recon['image'])
    return image_list

    


# ### Transformation Utils ###
def apply_transform(geom: Union[Meshes, Pointclouds, torch.Tensor], trans: Transform3d):
    if not isinstance(trans, Transform3d):
        if trans.ndim == 2:
            trans = geom_utils.se3_to_matrix(trans)
        trans = Transform3d(matrix=trans.transpose(1, 2), device=trans.device)
    verts = get_verts(geom)
    verts = trans.transform_points(verts)
    if hasattr(geom, 'update_padded'):
        geom = geom.update_padded(verts)
    else:
        geom = verts
    return geom


def center_obj(geom: Union[Meshes, Pointclouds]):
    """ Move center of geom to (0,0,0)
    :return Meshes, center: (N, 3)"""
    points = get_verts(geom)
    center = torch.sum(points, dim=1, keepdim=True)  # (N, 1, 3)
    num_verts = get_num_verts(geom)
    center = center / num_verts.unsqueeze(-1).unsqueeze(-1)  # N, 1, 3

    verts = points - center  # (N, P, 3)
    geom = geom.update_padded(verts)

    return geom, center.squeeze(1)


def transform_verts_to_view(geom: Union[Meshes, Pointclouds], cameras: PerspectiveCameras):
    src_trans = cameras.get_world_to_view_transform()
    view_point = get_verts(geom)
    view_points = src_trans.transform_points(view_point)
    dst_geom = geom.update_padded(view_points)
    return dst_geom


def scale_geom(geom, scale):
    """
    :param geom: meshes or pc (N, P, 3+C)
    :param scale: (N, 1/3)
    """
    geom = geom.clone()
    if not torch.is_tensor(scale):
        scale = torch.tensor([[scale]], dtype=torch.float32, device=geom.device)
    if isinstance(geom, Meshes) or isinstance(geom, pytorch3d.structures.Meshes):
        scale_trans = Scale(scale.expand(scale.size(0), 3), device=scale.device)
        geom = apply_transform(geom, scale_trans)
    elif torch.is_tensor(geom):
        xyz, feat = geom.split([3, geom.size(-1) - 3], dim=-1)
        geom = torch.cat([xyz * scale.unsqueeze(1), feat], dim=-1)
    else:
        raise NotImplementedError
    return geom


# ######## PointCloud / Meshes Utils ########
def get_num_verts(geom: Union[Meshes, Pointclouds]):
    if isinstance(geom, Meshes) or isinstance(geom, pytorch3d.structures.Meshes):
        num = geom.num_verts_per_mesh()
    elif isinstance(geom, Pointclouds):
        num = geom.num_points_per_cloud()
    else:
        raise NotImplementedError
    return num


def get_verts(geom: Union[Meshes, Pointclouds]) -> torch.Tensor:
    if isinstance(geom, Meshes) or isinstance(geom, pytorch3d.structures.Meshes):
        view_points = geom.verts_padded()
    elif isinstance(geom, Pointclouds):
        view_points = geom.points_padded()
    elif isinstance(geom, torch.Tensor):
        view_points = geom
    else:
        raise NotImplementedError(type(geom))
    return view_points


def get_render_func(geom: Union[Meshes, Pointclouds]):
    if isinstance(geom, Meshes) or isinstance(geom, pytorch3d.structures.Meshes):
        func = render_mesh
    elif isinstance(geom, Pointclouds):
        func = render_pc
    else:
        raise NotImplementedError
    return func


# ######## Detpth & Surface Normal Utils ########
def rdn_pixel_to_pc(xyz_field, prob_mask, image=None, nP=1024):
    """
    :param xyz_field: (N, C, H, W)
    :param prob_mask: (N, 1, H, W)
    :param image: (N, D, h, w)
    :param nP:
    :return: (N, P, C+D), (N, P, 2)
    """
    N, C, H, W = xyz_field.size()
    inds = torch.multinomial(prob_mask.view(N, H * W), nP).unsqueeze(1)  # (N, 1, P)

    xyz_points = torch.gather(xyz_field.view(N, C, H * W), dim=-1, index=inds.expand(N, C, nP), )
    # todo: test
    y = inds // W
    x = inds % W

    x, y = x.float(), y.float()
    x = 2 * x / W - 1
    y = 2 * y / H - 1
    grid = torch.stack([x, y], dim=-1)  # (N, 1, P, 2)

    if image is not None:
        _, D, iH, iW = image.size()
        if iH != H:
            image_points = F.grid_sample(image, grid)  # (N, D, 1, P)
            image_points = image_points.view(N, D, nP)
        else:
            image_points = torch.gather(image.view(N, D, H * W), dim=-1, index=inds.expand(N, D, nP))
        xyz_points = torch.cat([xyz_points, image_points], 1)

    xyz_points = xyz_points.transpose(-1, -2).contiguous()
    return xyz_points, grid.squeeze(1)


def unproj_depth_to_xyz(depth: torch.Tensor, camera: PerspectiveCameras = None, scale=True, xy=None) -> torch.Tensor:
    """
    lift a depth map to point cloud
    :param torch Tensor of depth: (N, (1,) H, W)
    :return: point cloud: (N, H, W, 3) in xyz World(camera) space?
    """
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    N, H, W = depth.size()
    device = depth.device

    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W)
    )  # (H, W)
    ys = ys.unsqueeze(0).expand(N, H, W).to(device)
    xs = xs.unsqueeze(0).expand(N, H, W).to(device)

    # if camera is not None:
    #     f, _ = get_camera_f_p(camera)
    # normalize depth
    if scale:
        # scale to [1, 2]??
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth + 1

    if xy is None:
        xy = torch.stack([xs, ys], -1)
    xydepth = torch.cat([xy, depth.unsqueeze(-1)], dim=-1)  # N, H, W, 3
    xydepth = xydepth.view(N, H * W, 3)
    XYZ = camera.unproject_points(xydepth,)  # (N, H*W, 3)
    XYZ = XYZ.view(N, H, W, 3)
    return XYZ


def xyz_to_pc(cXyz, image: torch.Tensor = None, mask: torch.Tensor = None, return_pc=True,
              permute_xyz=False) -> Pointclouds:
    """
    :param cXyz: (N, H, W, 3) in camera frame
    :param image: (N, C, H, W)
    :param mask: (N, 1, H, W)
    :return: (N, P, 3), (N, P, 3)
    """
    device = cXyz.device

    if permute_xyz:
        cXyz = cXyz.permute(0, 2, 3, 1)
    N, H, W, _ = cXyz.size()

    if mask is None:
        mask = torch.ones([N, H, W], device=device)
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    mask = (mask > 0.5)

    if image is None:
        image = torch.ones([N, 3, H, W], device=device) / 3 * 2
    color = image.permute(0, 2, 3, 1)
    C = color.size(-1)

    cXyz = torch.masked_select(cXyz, mask.unsqueeze(-1)).view(-1, 3)  # Nx * 3
    color = torch.masked_select(color, mask.unsqueeze(-1)).view(-1, C)

    num_point = mask.view(N, -1).long().sum(-1).tolist()  # number of point (N, )
    cXyz = torch.split(cXyz, num_point, dim=0)  # [(Ni, 3), ]
    color = torch.split(color, num_point, dim=0)

    if return_pc:
        pc = Pointclouds(list(cXyz), features=list(color)).to(device)
        return pc
    else:
        return xyz, color


def depth_to_pc(depth, image: torch.Tensor=None, cameras=None, mask=None, scale=False, return_pc=True) -> Pointclouds:
    """
    :param depth: (N, H, W)
    :param image: (N, C, H, W)
    :param cameras:
    :param mask: (N, H, W)
    :return: (N, P, 3), (N, P, 3)
    """
    if mask is None:
        mask = torch.ones_like(depth)
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    mask = mask.bool()
    xyz = unproj_depth_to_xyz(depth, cameras, scale)  # N, H, W, 3

    return xyz_to_pc(xyz, image, mask, return_pc)



# ######## Camera utils ########
def weak_to_full_persp(f, pp, scale, trans):
    """
    Args:
        f ([type]): (N, (1))
        pp ([type]): [description]
        scale ([type]): [description]
        trans ([type]): [description]
    Returns:
        (X,Y,Z) + (px / scale, py / scale, f / scale)  (N, 3)
    """
    if f.ndim == 1:
        f = f.unsqueeze(-1)
    if f.size(-1) == 2:
        f = f[..., 0:1]
        logging.warn('does not support fx fy')
    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)

    # trans = torch.cat([trans - pp, f], dim=-1)  # N, 3
    # trans = trans / scale  # N, 3
    translate = torch.cat([trans - pp / scale, f / scale], -1)
    return translate

def intr_from_ndc_to_screen(ndc_intr, H, W):
    """-1, 1 --> 0, H, it's essentially affine transformation.... 
    """
    N = len(ndc_intr)
    device = ndc_intr.device
    scale_mat = torch.FloatTensor([[
        [W/2, 0, W/2, 0],
        [0, H/2, H/2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]]).to(device).repeat(N, 1, 1)
    pix_intr = scale_mat @ ndc_intr
    return pix_intr


def intr_from_screen_to_ndc(pix_intr, H, W):
    """0,H --> -1, 1"""
    N = len(pix_intr)
    device = pix_intr.device
    scale_mat = torch.FloatTensor([[
        [2/W, 0, -1, 0],
        [0, 2/H, -1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]]).to(device).repeat(N, 1, 1)
    ndc_intr = scale_mat @ pix_intr
    return ndc_intr


def get_fxfy_pxpy(K):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    px = K[..., 0, 2]
    py = K[..., 1, 2]
    f = torch.stack([fx, fy], -1)
    p = torch.stack([px, py], -1)
    return f, p


def weak2full_perspective(meshes: Meshes, scale, trans, dst_camera: PerspectiveCameras):
    """
    :param meshes:
    :param scale: (N, )
    :param trans: (N, 2) px, py in screen space [-1, 1]
    :param dst_camera: full perspective camera with focial length f (N,)
    :return: (X,Y,Z) + (px / scale, py / scale, f / scale)
    """
    f, pp = get_camera_f_p(dst_camera)
    trans = torch.cat([trans - pp, f.unsqueeze(-1)], dim=-1)  # N, 3
    trans = trans / scale.unsqueeze(-1)  # N, 3
    trans = trans.unsqueeze(1)  # N, 1, 3
    meshes = meshes.update_padded(meshes.verts_padded() + trans)
    return meshes


def param_to_cameras(azel, xyz, cameras: PerspectiveCameras):
    """
    :param param: (N, 5)
    :param cameras: reuses its intrinsic if any
    :return:
    """
    if cameras is not None:
        f, p = get_camera_f_p(cameras)
        # k = cameras.get_projection_transform().get_matrix()
        # f = torch.stack([k[:, 0, 0], k[:, 1, 1]], dim=1)
        # p = k[:, 0:2, 2]
    else:
        f = 10;
        p = ((0.0, 0.0),)

    R = geom_utils.azel_to_rot(azel, homo=False)
    # cameras = PerspectiveCameras(f, p, R, xyz, device=azel.device)
    cameras = PerspectiveCameras(f, p, R, xyz, device=azel.device)
    return cameras


def get_k(camera: PerspectiveCameras):
    """
    :param camera:
    :return: (N, 4, 4)
    """
    K = camera.get_projection_transform().get_matrix().transpose(-1, -2)
    K[..., 2, 2] = K[..., 3, 3] = 1
    K[..., 2, 3] = K[..., 3, 2] = 0
    return K

def get_k_from_fp(f, p):
    device = f.device
    N = len(f)
    K = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
    K[:, 0, 0] = f[:, 0]
    K[:, 1, 1] = f[:, 1]
    K[:, 0:2, 2] = p
    return K

def f_to_K(f):
    """
    Args:
        f ([type]): (N, 2)
    return: (N, 4, 4)
    """
    diag = torch.cat([f, torch.ones_like(f)], dim=-1)
    K = torch.diag_embed(diag, )
    return K

def get_camera_dist(cTw=None, wTc=None):
    """
    Args:
        cTw (N, 4, 4) extrinsics

    Returns:
        (N, )
    """
    if wTc is None:
        wTc = geom_utils.inverse_rt(mat=cTw, return_mat=True)
    cam_norm = wTc[..., 0:4, 3]
    cam_norm = cam_norm[..., 0:3] / cam_norm[..., 3:4]  # (N, 3)
    norm = torch.norm(cam_norm, dim=-1)
    return norm
    
def get_camera_f_p(cameras: PerspectiveCameras):
    """(N, ), (N, 2)"""
    proj = cameras.get_projection_transform().get_matrix()
    return proj[:, 0, 0], proj[:, 0:2, 2]


def get_principal_axis(cameras: PerspectiveCameras):
    """(N, 3)"""
    M = cameras.get_projection_transform().get_matrix().transpose(1, 2)  # m^3: (N, 4, 4)
    direction = M[:, 3, 0: 3]
    return direction


def get_view_list(view_mod, device='cpu', time_len=21, **kwargs):
    """
    :param view_mod:
    :param x:
    :return: (T, 2) : azel, xyz
    """
    view_dict = {
        'az': (0, np.pi * 2, 10 / 180 * np.pi),
        'el': (-np.pi / 2, np.pi / 2, 5 / 180 * np.pi),
    }
    zeros = torch.zeros([time_len])
    if 'az' in view_mod:
        vary_az = torch.linspace(view_dict[view_mod][0], view_dict[view_mod][1], time_len)
        vary_el = zeros
    elif 'el' in view_mod:
        vary_az = zeros
        vary_el = torch.linspace(view_dict[view_mod][0], view_dict[view_mod][1], time_len)
    elif 'circle' in view_mod:
        theta = torch.linspace(0, np.pi * 2, time_len)
        cone = 2/3
        vary_az = cone * torch.cos(theta)
        vary_el = cone * torch.sin(theta)
    else:
        raise NotImplementedError

    azel = torch.stack([vary_az, vary_el], dim=1).to(device)  # (T, 2)
    return azel


def ambient_light(device='cpu', cameras: PerspectiveCameras = None, **kwargs):
    d = torch.FloatTensor([[0, 0, -1]]).to(device)
    N = 1 if cameras is None else len(cameras)
    zeros = torch.zeros([N, 3], device=device)
    d = zeros + d
    if cameras is not None:
        d = cameras.get_world_to_view_transform().inverse().transform_normals(d.unsqueeze(1))
        d = d.squeeze(1)

    color = kwargs.get('light_color', np.array([0.65, 0.3, 0.0]))
    am, df, sp = color
    am = zeros + am
    df = zeros + df
    sp = zeros + sp
    lights = DirectionalLights(
        device=device,
        ambient_color=am,
        diffuse_color=df,
        specular_color=sp,
        direction=d,
    )
    return lights


def local_to_world_mesh(meshes: Meshes, trans):
    """
    :param meshes:
    :param trans: ([1], 4, 4) or numpy. world = trans * local_verts
    :return:
    """
    if not torch.is_tensor(trans):
        trans = torch.FloatTensor(trans, device=meshes.device)
    if trans.ndimension() == 2:
        trans = trans.unsqueeze(0)

    verts = meshes.verts_padded()
    N, V, _ = verts.size()

    # put objet mesh to world frame
    if trans.size(-1) == 4:
        # homo
        trans = Transform3d(device=trans.device, matrix=trans.transpose(1, 2))
        # verts = torch.cat([verts, torch.ones([N, V, 1])], dim=-1)  # N, V, 4
        # new_verts = torch.matmul(verts, trans.transpose(1, 2))[:, :, 0:3]
        new_verts = trans.transform_points(verts)
    elif trans.size(-1) == 3:
        new_verts = torch.matmul(verts, trans.transpose(1, 2))
    meshes = meshes.update_padded(new_verts)
    return meshes


def homo_np(pose):
    """(H, W)"""
    H, W = pose.shape
    if H == 4 and W == 4:
        eye = pose
    else:
        eye = np.eye(4)
        eye[0:H, 0:W] = pose
    return eye


def detect_contact(j3d, v3d, th):
    """if exists contact"""
    if torch.is_tensor(j3d):
        torch.cdist(j3d, v3d)
    else:
        all_dists = cdist(j3d, v3d)
    return all_dists.min() <= th


def np_to_batch_tensor(np_tensor, device='cpu', type='float', add_b_dim=False):
    if torch.is_tensor(np_tensor):
        tensor = np_tensor
    else:
        np_tensor = np.array(np_tensor)
        tensor = torch.FloatTensor(np_tensor).to(device)
    if type == ' int':
        tensor = tensor.long()
    if add_b_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


# ######## Batch utils ########
def join_scene(mesh_list: List[Meshes]) -> Meshes:
    """Joins a list of meshes to single Meshes of scene"""
    # simple check
    if len(mesh_list) == 1:
        return mesh_list[0]
    device = mesh_list[0].device

    v_list = []
    f_list = []
    t_list = []
    for m, mesh in enumerate(mesh_list):
        v_list.append(mesh.verts_list())
        f_list.append(mesh.faces_list())
        if mesh.textures is None or isinstance(mesh.textures, TexturesUV):
            mesh.textures = pad_texture(mesh)
        t_list.append(mesh.textures.verts_features_list())
        N = len(mesh)

    scene_list = []
    for n in range(N):
        verts = [v_list[m][n] for m in range(len(mesh_list))]
        faces = [f_list[m][n] for m in range(len(mesh_list))]
        texes = [t_list[m][n] for m in range(len(mesh_list))]
        # merge to one scene
        tex = TexturesVertex(texes)
        mesh = Meshes(verts, faces, textures=tex).to(device)
        tex = TexturesVertex(mesh.textures.verts_features_packed().unsqueeze(0))
        scene = Meshes(verts=mesh.verts_packed().unsqueeze(0), faces=mesh.faces_packed().unsqueeze(0), textures=tex)

        scene_list.append(scene)
    scene = join_meshes_as_batch(scene_list, include_textures=True)
    # texture = TexturesVertex(torch.ones_like(verts))
    # scene.textures = texture

    return scene

def expand_meshes(meshes: Meshes, N):
    assert len(meshes) == 1
    

def join_scene_w_labels(mesh_list: List[Meshes], num_classes=3) -> Meshes:
    assert len(mesh_list) <= num_classes
    for m, mesh in enumerate(mesh_list):
        N = len(mesh)
        device = mesh.device
        feat = torch.zeros([mesh.verts_packed().size(0), num_classes], device=device)
        feat[..., m] = 1
        new_mesh = Meshes(mesh.verts_list(), mesh.faces_list(), pad_texture(mesh, feat))
        mesh_list[m] = new_mesh
    return join_scene(mesh_list)


def join_homo_meshes(verts, faces, textures=None):
    """ assume dim=1
    :param verts: (N, K, V, 3)
    :param faces: (N, K, F, 3)
    :param textures: (N, K, V, 3)
    :param dim:
    :return:
    """
    N, K, V, _ = verts.size()
    F = faces.size(2)
    device = verts.device

    verts = verts.view(N, K * V, 3)
    off_faces = (torch.arange(0, K, device=device) * V).view(1, K, 1, 1)
    faces = (off_faces + faces).view(N, K * F, 3)

    if textures is None:
        textures = torch.ones_like(verts)
    else:
        textures = textures.view(N, K * V, 3)
    return Meshes(verts, faces, TexturesVertex(textures))


### Voxel Utils ####
def cubify(voxels, th=0.5):
    """
    Args:
        voxels ([type]): (N, 1, D, H, W)
        th (float, optional): [description]. Defaults to 0.5.
    Returns:
        [type]: Mesehs, in [-1, 1]
    """
    meshes = ops_3d.cubify(voxels.squeeze(1), th,  align='corner')
    meshes.textures = pad_texture(meshes, torch.ones_like(meshes.verts_padded()))
    return meshes
    

### SDF Utils #####
def batch_sdf_to_meshes(sdf: Callable, batch_size, total_max_batch=32 ** 3, bound=False, **kwargs):
    """convert a batched sdf to meshes
    Args:
        sdf (Callable): signature: sdf(points (N, P, 3), **kwargs) where kwargs should be filled 
        batch_size ([type]): batch size in **kwargs
        total_max_batch ([type], optional): [description]. Defaults to 32**3.
    Returns:
        Mehses
    """
    N = kwargs.get('N', 64)
    samples, voxel_origin, voxel_size = grid_xyz_samples(N)  # (P, 3) on cpu
    samples = samples.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, P, 3)
    num_samples = samples.size(1)

    head = 0
    max_batch = total_max_batch // batch_size
    while head < num_samples:
        sample_subset = samples[:, head: min(head + max_batch, num_samples), 0:3].cuda()
        samples[:, head: min(head + max_batch, num_samples), 3:4] = sdf(sample_subset).detach().cpu()
        head += max_batch

    sdf_values = samples[..., 3]  # (B, N*N*N)
    sdf_values = sdf_values.reshape(batch_size, N, N, N)


    verts_list, faces_list, tex_list = [], [], []
    for n in range(batch_size):
        # marching cube
        verts, faces = convert_sdf_samples_to_ply(sdf_values[n].cpu(), voxel_origin, voxel_size, add_bound=bound)
        device = samples.device
        verts = verts.to(device)
        faces = faces.to(device)
        verts_list.append(verts)
        faces_list.append(faces)
        tex_list.append(torch.ones_like(verts))
    meshes = Meshes(verts_list, faces_list).cuda()
    if meshes.isempty():
        meshes.textures = TexturesVertex(torch.ones([batch_size, 0, 3]).cuda())
    return meshes


def sdf_to_meshes(sdf: Callable, cat_func: Callable=lambda x:x, z=None, **kwargs):
    verts_list, faces_list, tex_list = [], [], []
    batch_size = z.size(0)
    for n in range(batch_size):
        func_z = functools.partial(cat_func, z=z[n:n + 1])
        func = lambda x: sdf(func_z(x))[:, 0]
        verts, faces = create_mesh(func, None, **kwargs)
        verts_list.append(verts)
        faces_list.append(faces)
        tex_list.append(torch.ones_like(verts))
    meshes = Meshes(verts_list, faces_list, TexturesVertex(tex_list)).cuda()
    return meshes

def grid_xyz_samples(N=64):
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate. convert to range [-1, 1]
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False
    return samples, voxel_origin, voxel_size

def create_mesh(
        decoder, filename=None, N=64, max_batch=32 ** 3, offset=None, scale=None,
        **kwargs
):
    """
    :param decoder:
    :param latent_vec:
    :param filename:
    :param N:
    :param max_batch:
    :param offset: (3, )
    :param scale:  float or (1, )
    :return: verts: Tensor in shape of (V, 3), faces: Tensor in shape of (F, 3)
    """
    start = time.time()
    ply_filename = filename

    # decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate. convert to range [-1, 1]
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]


    if scale is not None:
        scale = scale.view(1, 1).cpu()
        samples[:, 0:3] = samples[:, 0:3] * scale
    if offset is not None:
        offset = offset.view(1, 3).cpu()
        samples[:, 0:3] = samples[:, 0:3] + offset

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0
    # assert False
    while head < num_samples:
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()
        samples[head: min(head + max_batch, num_samples), 3] = \
            decoder(sample_subset).detach().cpu()
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    offset = offset.detach().numpy() if offset is not None else None
    scale = scale.detach().numpy() if scale is not None else None
    verts, faces = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename,
        offset,
        scale,
    )
    device = samples.device
    verts = verts.to(device)
    faces = faces.to(device)
    return verts, faces

def convert_sdf_samples_to_ply(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        ply_filename_out=None,
        offset=None,
        scale=None,
        add_bound=False,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    :return torch.FloatTensor of shape (V, 3), (F, 3)
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    if add_bound:
        N = numpy_3d_sdf_tensor.shape[0]
        voxel_size = voxel_size * (N - 1) / (N + 1)
        numpy_3d_sdf_tensor = np.pad(numpy_3d_sdf_tensor, 1, constant_values=1.)
        # voxel_size = 2.0 / (N - 1)

        # if scale is None:
        #     scale = 1
        # scale = scale * (voxel_size + 2) / (voxel_size)
    
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
        # verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        #     numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        # )
    except ValueError:
        return torch.empty([0, 3], dtype=torch.float32), torch.empty([0, 3], dtype=torch.long)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # # apply additional offset and scale
    if scale is not None:
        # mesh_points = mesh_points / scale
        mesh_points = mesh_points * scale
    if offset is not None:
        # mesh_points = mesh_points - offset
        mesh_points = mesh_points + offset

    verts_tensor = torch.from_numpy(np.array(mesh_points)).float()
    faces_tensor = torch.from_numpy(np.array(faces)).long()

    return verts_tensor, faces_tensor




# #### collate function ####


def collate_meshes(batch):
    """
    collate function specifiying Meshes
    :return:
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")


    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, Meshes) or isinstance(elem, pytorch3d.structures.Meshes):
        meshes = join_meshes_as_batch(batch)
        return meshes
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_meshes([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return  {key: collate_meshes([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_meshes(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_meshes(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))



def test_end_fscore(outputs, save_dir, suf='', head=''):
    mean_list = []
    fname = osp.join(save_dir, 'f1_num%s.txt' % suf)
    os.makedirs(save_dir, exist_ok=True)
    with open(fname, 'w') as fp:
        fp.write('%s\n' % head)
    num_metric = len(outputs[0])
    
    sample_list = [[] for _ in range(num_metric)]
    for batch_res in outputs:
        for c, batch_metric in enumerate(batch_res):
            sample_list[c].extend(batch_metric)
    for r in range(len(sample_list[0])):
        with open(fname, 'a') as fp:
            fp.write('%s\n' % ','.join(str(sample_list[c][r]) 
                for c in range(len(sample_list))))
    mean_list = []
    # sample_list = np.array(sample_list)
    for c in range(num_metric):
        if isinstance(sample_list[c][0] , str):
            mean_list.append('NA')
        else:
            mean_list.append(str(np.mean(sample_list[c])))

    with open(fname , 'a') as fp:
        fp.write('mean,%s\n' % ','.join('%s' % f for f in mean_list))
        print('mean,%s\n' % ','.join('%s' % f for f in mean_list))
    print('save to %s' % save_dir)
    return mean_list
        

def iou(pred, gt, thresh=.5, reduction='none'):
    """
    :param pred: numpy in N, ...
    :param gt: numpy in N, ...
    :param thresh:
    :return:
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().detach().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().detach().numpy()

    pred = pred.copy()
    N = pred.shape[0]
    pred[np.where(pred > thresh)] = 1
    pred[np.where(pred <= thresh)] = 0

    pred = np.reshape(pred, [N, -1])
    gt = np.reshape(gt, [N, -1])
    iou = np.sum(pred * gt, axis=1) / np.sum(np.clip(pred + gt, a_min=0, a_max=1), axis=1)
    assert iou.shape[0] == N
    if reduction == 'mean':
        iou = np.mean(iou)
    return iou.tolist()


def fscore(pred_pts: Meshes, gt_pts: Meshes, trans=None, num_samples=10000, th=None):
    """
    Calculate bathced F-score between  2 point clouds
    https://github.com/lmb-freiburg/what3d/blob/1f8e6bf5b1334166b02d9b86b14354edb77992bd/util.py#L54
    """
    if th is None:
        th_list = [.2/100, .5/ 100, 1./100, 1./50, 1./20, 1./10]
    else:
        th_list = th
    if isinstance(pred_pts, Meshes) or isinstance(pred_pts, pytorch3d.structures.Meshes):
        if pred_pts.isempty():
            pred_pts = torch.zeros([len(gt_pts), num_samples, 3]).to(pred_pts.device) + 10000
        else:
            pred_pts = ops_3d.sample_points_from_meshes(pred_pts, num_samples)  # N, K, 3

    if isinstance(gt_pts, Meshes) or isinstance(gt_pts, pytorch3d.structures.Meshes):
        gt_pts = ops_3d.sample_points_from_meshes(gt_pts, num_samples)
    
    (d1, d2), _ = chamfer_distance(pred_pts, gt_pts, batch_reduction=None, point_reduction=None)
    d1 = torch.sqrt(d1)
    d2 = torch.sqrt(d2)

    res_list = []
    for th in th_list:
        if d1.size(1) and d2.size(1):
            recall = torch.sum(d2 < th, dim=-1).to(gt_pts) / num_samples  # recall knn(gt, pred) gt->pred
            precision = torch.sum(d1 < th, dim=-1).to(gt_pts) / num_samples  # precision knn(pred, gt) pred-->

            eps = 1e-6
            fscore = 2 * recall * precision / (recall + precision + eps)
            # res_list.append([fscore, precision, recall])
            res_list.append(fscore.tolist())
        else:
            raise ValueError("d1 and d2 should be in equal length but got %d %d" % (d1.size(1), d2.size(1)))
    d = ((d1 ** 2).mean(1) + (d2 ** 2).mean(1)).tolist()
    return res_list + [d, ]

def cdscore(pred_pts: Meshes, gt_pts: Meshes, trans=None, num_samples=10000, th=None):
    if isinstance(pred_pts, Meshes) or isinstance(pred_pts, pytorch3d.structures.Meshes):
        if pred_pts.isempty():
            pred_pts = torch.zeros([len(gt_pts), num_samples, 3]).to(pred_pts.device) + 10000
        else:
            pred_pts = ops_3d.sample_points_from_meshes(pred_pts, num_samples)  # N, K, 3

    if isinstance(gt_pts, Meshes) or isinstance(gt_pts, pytorch3d.structures.Meshes):
        gt_pts = ops_3d.sample_points_from_meshes(gt_pts, num_samples)
    
    (d1, d2), _ = chamfer_distance(pred_pts, gt_pts, batch_reduction=None, point_reduction=None)
    d = ((d1).mean(1) + (d2).mean(1)).tolist()
    return d

def to_trimesh(meshes: Meshes) -> Trimesh:
    assert len(meshes) == 1
    vert = meshes.verts_list()[0].cpu().detach().numpy()
    face = meshes.faces_list()[0].cpu().detach().numpy()
    return Trimesh(vert, face)


def voxelize(mesh: trimesh.Trimesh, reso=32, return_torch=False) -> Tuple[np.ndarray, VoxelGrid]:
    """
    Args:
        mesh (trimesh.Trimesh): [description]
        reso (int, optional): [description]. Defaults to 32.
        return_torch (bool, optional): [description]. Defaults to False.
    Returns:
        if return_torch = True: (1, D, H, W)
        Tuple[np.ndarray, VoxelGrid]: [description]
    """
    x_l = torch.linspace(-1, 1, reso)
    y_l = torch.linspace(-1, 1, reso)
    z_l = torch.linspace(-1, 1, reso)
    z, y, x = torch.meshgrid(z_l, y_l, x_l)
    points_cords = torch.stack([x,y,z], dim=-1)
    
    if isinstance(mesh, trimesh.Trimesh):
        voxel_mesh = mesh.voxelized(pitch=2/reso)
    elif isinstance(mesh, Meshes) or isinstance(mesh, pytorch3d.structures.Meshes):
        mesh = to_trimesh(mesh)
        voxel_mesh = mesh.voxelized(pitch=2/reso)
    else:
        voxel_mesh = mesh
    
    points_cords = points_cords.detach().cpu().numpy()
    points_cords = points_cords.reshape(-1,3)
    voxels = voxel_mesh.is_filled(points_cords)
    voxels = voxels.reshape(reso, reso, reso)

    if return_torch:
        voxels = torch.FloatTensor([voxels])
    return voxels, voxel_mesh


def sample_unit_cube(hObj, num_points, r=1):
        """
        Args:
            points (P, 4): Description
            num_points ( ): Description
            r (int, optional): Description
        
        Returns:
            sampled points: (num_points, 4)
        """
        D = hObj.size(-1)
        points = hObj[..., :3]
        prob = (torch.sum((torch.abs(points) < r), dim=-1) == 3).float()
        if prob.sum() == 0:
            prob = prob + 1
            print('oops')
        inds = torch.multinomial(prob, num_points, replacement=True).unsqueeze(-1)  # (P, 1)

        handle = torch.gather(hObj, 0, inds.repeat(1, D))
        return handle

class SoftFlatShader(nn.Module):
    """
    Per face lighting - the lighting model is applied using the average face
    position and the face normal. The blending function hard assigns
    the color of the closest face for each pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = HardFlatShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = flat_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images

if __name__ == '__main__':
    from nnutils import image_utils
    import numpy as np

    device = 'cuda:0'
    mat = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    meshes = gripper_mesh(mat=mat)
    save_dir = '/glusterfs/yufeiy2/hoi_output/vis/'
    os.makedirs(save_dir, exist_ok=True)

    # image_list = render_geom_rot(meshes, scale_geom=True)
    # image_utils.save_gif(image_list, os.path.join(save_dir, 'gripper_s.png'))
    # image_list = render_geom_rot(meshes)
    # image_utils.save_gif(image_list, os.path.join(save_dir, 'gripper.png'))

    for index in range(2):
        pkl = np.load(os.path.join(save_dir, 'grasp_%d.npz' % index))
        obj = np.array(pkl['obj'])
        N = obj.shape[0];
        T = np.prod(obj.shape) // N // 9
        grasp = np.array(pkl['grasp'])  # 8, (n,), 9

        obj = torch.tensor(obj, dtype=torch.float32, device=device)
        grasp = torch.tensor(grasp, dtype=torch.float32, device=device)

        xyz, feature = obj.split([3, 3], dim=-1)
        obj_meshes = pc_to_cubic_meshes(xyz, feature / 255 * 2 - 1)
        image_utils.save_gif(render_geom_rot(obj_meshes, scale_geom=True), '%s/%d_obj.png' % (save_dir, index))
        gripper = gripper_mesh(grasp)
        image_utils.save_gif(render_geom_rot(gripper, scale_geom=True), '%s/%d_gripper.png' % (save_dir, index))

        scene = join_scene([obj_meshes, gripper])
        image_utils.save_gif(render_geom_rot(scene, scale_geom=True), '%s/%d_scene.png' % (save_dir, index))