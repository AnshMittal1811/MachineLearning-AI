import pyrender
import trimesh
import numpy as np
import os
import platform
from config import *
from src.dataset.uv_face import uvm2mesh
from PIL import Image

if platform.system() != 'Windows':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

r = pyrender.OffscreenRenderer(CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE)
scene = pyrender.Scene()


def render_face_orthographic(mesh, background=None):
    """
    mesh location should be normalized
    :param mesh:
    :param background:
    :return:
    """
    mesh.visual.face_colors = np.array([0.05, 0.1, 0.2, 1])

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # mesh = pyrender.Mesh.from_trimesh(mesh)

    scene.add(mesh, pose=np.eye(4))
    camera_pose = np.eye(4)
    # camera_pose[0, 3] = 1
    # camera_pose[1, 3] = 1
    # camera_pose[2, 3] = -10
    # camera_pose[0, 0] = 1
    # camera_pose[1, 1] = -1
    # camera_pose[2, 2] = -1
    #
    # camera = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=100)
    camera_pose[0, 3] = 1
    camera_pose[1, 3] = 1
    camera_pose[2, 3] = 10
    camera_pose[0, 0] = 1
    camera_pose[1, 1] = 1
    camera_pose[2, 2] = 1

    camera = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=100)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene)
    scene.clear()

    # print(color.shape)
    color = np.array(color)
    color = color[::-1]
    if background is not None:
        new_color = np.array(background)
        new_color[color != 255] = color[color != 255]
        color = new_color
    return color


def render_face_transparent(mesh, background=None):
    """
    mesh location should be normalized
    :param mesh:
    :param background:
    :return:
    """
    mesh.visual.face_colors = np.array([0.5, 0.5, 0.5, 1])  # np.array([0.05, 0.1, 0.2, 1])

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # mesh = pyrender.Mesh.from_trimesh(mesh)

    scene.add(mesh, pose=np.eye(4))
    camera_pose = np.eye(4)
    # camera_pose[0, 3] = 1
    # camera_pose[1, 3] = 1
    # camera_pose[2, 3] = -10
    # camera_pose[0, 0] = 1
    # camera_pose[1, 1] = -1
    # camera_pose[2, 2] = -1
    #
    # camera = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=100)
    camera_pose[0, 3] = 1
    camera_pose[1, 3] = 1
    camera_pose[2, 3] = 10
    camera_pose[0, 0] = 1
    camera_pose[1, 1] = 1
    camera_pose[2, 2] = 1

    camera = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=100)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    scene.clear()

    # print(color.shape)
    color = np.array(color)
    color = color[::-1]
    if background is not None:
        bg_rgba = Image.fromarray(np.array(background)).convert('RGBA')

        color[color[:, :, 0] == 255, 3] = 0
        color[color[:, :, 3] == 255, 3] = 166

        color = Image.fromarray(color).convert('RGBA')
        final = Image.alpha_composite(bg_rgba, color)
        color = np.array(final)

        #
        # new_color = np.array(background)
        # new_color[color != 255] = color[color != 255]
        # color = new_color

    return color


def render_uvm(face_uvm, img=None):
    face_mesh = uvm2mesh(face_uvm / CROPPED_IMAGE_SIZE * 2)
    ret = render_face_orthographic(face_mesh, img)
    return ret
