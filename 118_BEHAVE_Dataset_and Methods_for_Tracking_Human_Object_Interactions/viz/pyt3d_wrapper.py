"""
a simple wrapper for pytorch3d rendering
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import numpy as np
import torch
from copy import deepcopy
# Data structures and functions for rendering
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PerspectiveCameras,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from viz.contact_viz import ContactVisualizer

SMPL_OBJ_COLOR_LIST = [
        [0.65098039, 0.74117647, 0.85882353],  # SMPL
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
    ]


class MeshRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"
    def __init__(self, image_size=1200,
                 faces_per_pixel=1,
                 device='cuda:0',
                 blur_radius=0, lights=None,
                 materials=None, max_faces_per_bin=50000):
        self.image_size = image_size
        self.faces_per_pixel=faces_per_pixel
        self.max_faces_per_bin=max_faces_per_bin # prevent overflow, see https://github.com/facebookresearch/pytorch3d/issues/348
        self.blur_radius = blur_radius
        self.device = device
        self.lights=lights if lights is not None else PointLights(
            ((0.5, 0.5, 0.5),), ((0.5, 0.5, 0.5),), ((0.05, 0.05, 0.05),), ((0, -2, 0),), device
        )
        self.materials = materials
        self.renderer = self.setup_renderer()

    def setup_renderer(self):
        # for sillhouette rendering
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=self.blur_radius,
            # blur_radius=np.log(1. / 1e-4 - 1.) * sigma, # this will create large sphere for each face
            faces_per_pixel=self.faces_per_pixel,
            clip_barycentric_coords=False,
            max_faces_per_bin=self.max_faces_per_bin
        )
        shader = SoftPhongShader(
            device=self.device,
            lights=self.lights,
            materials=self.materials)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings),
                shader=shader
        )
        return renderer

    def render(self, meshes, cameras, ret_mask=False):
        images = self.renderer(meshes, cameras=cameras)
        # print(images.shape)
        if ret_mask:
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :3].cpu().detach().numpy(), mask > 0
        return images[0, ..., :3].cpu().detach().numpy()


class Pyt3DWrapper:
    def __init__(self, image_size, device='cuda:0', colors=SMPL_OBJ_COLOR_LIST):
        self.renderer = MeshRendererWrapper(image_size, device=device)
        self.front_camera = self.get_kinect_camera(device)
        self.colors = deepcopy(colors)
        self.device = device
        self.contact_vizer = ContactVisualizer()

    @staticmethod
    def get_kinect_camera(device='cuda:0'):
        R, T = torch.eye(3), torch.zeros(3)
        R[0, 0] = R[1, 1] = -1 # pytorch3d y-axis up, need to rotate to kinect coordinate
        R = R.unsqueeze(0)
        T = T.unsqueeze(0)
        fx, fy = 979.7844, 979.840  # focal length
        cx, cy = 1018.952, 779.486  # camera centers
        color_w, color_h = 2048, 1536  # kinect color image size
        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)
        cam = PerspectiveCameras(focal_length=focal_length, principal_point=cam_center,
                                 image_size=((color_w, color_h),),
                                 device=device,
                                 R=R, T=T)
        return cam

    def render_meshes(self, meshes, viz_contact=False):
        """
        render a list of meshes
        :param meshes: a list of psbody meshes
        :return: rendered image
        """
        colors = deepcopy(self.colors)
        if viz_contact:
            contact_regions = self.contact_vizer.get_contact_spheres(meshes[0], meshes[1])
            for k, v in contact_regions.items():
                color, sphere = v
                meshes.append(sphere)
                colors.append(color)
        pyt3d_mesh = self.prepare_render(meshes, colors)
        rend = self.renderer.render(pyt3d_mesh, self.front_camera)
        return rend

    def prepare_render(self, meshes, colors):
        py3d_meshes = []
        for mesh, color in zip(meshes, colors):
            vc = np.zeros_like(mesh.v)
            vc[:, :] = color
            text = TexturesVertex([torch.from_numpy(vc).float().to(self.device)])
            py3d_mesh = Meshes([torch.from_numpy(mesh.v).float().to(self.device)], [torch.from_numpy(mesh.f.astype(int)).long().to(self.device)],
                               text)
            py3d_meshes.append(py3d_mesh)
        joined = join_meshes_as_scene(py3d_meshes)
        return joined





