import torch
import torchvision
import torch.nn.functional as F

import numpy as np

from shape_model.mesh_obj import mesh_obj
from PIL import Image

from pytorch3d.vis.plotly_vis import plot_scene,Lighting
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader,
    HardGouraudShader, SoftGouraudShader, HardPhongShader,
    TexturesVertex, TexturesUV, DirectionalLights
)

def renderfaces(verts, face_v, fp, device):
    verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
    verts_rgb[:,:,:,0] = verts_rgb[:,:,:,0]*255/255
    verts_rgb[:,:,:,1] = verts_rgb[:,:,:,1]*224/255
    verts_rgb[:,:,:,2] = verts_rgb[:,:,:,2]*189/255
    textures = TexturesVertex(verts_features=verts_rgb.squeeze().to(device))

    meshes = Meshes(verts=verts, faces=torch.cat(verts.shape[0]*[face_v]), textures=textures).to(device)
    R, T = look_at_view_transform(250, 0, 0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=True,
        )

    blend_params=BlendParams()
    blend_params._replace(background_color=(0.0,0.0,0.0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardGouraudShader(device=device, cameras=cameras, blend_params=blend_params)
        )

    images = renderer(meshes,cameras=cameras,lights=DirectionalLights(device=device, direction=((0,0,1),)))

    images = images.permute(0,3,1,2)
    torchvision.utils.save_image(images,fp=fp)
