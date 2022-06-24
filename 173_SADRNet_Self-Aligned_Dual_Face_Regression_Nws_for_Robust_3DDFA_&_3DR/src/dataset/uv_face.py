import src.faceutil
from src.faceutil.morphable_model import MorphabelModel
from PIL import Image
from config import *
import trimesh
import os

face_mask_np = np.array(Image.open(UV_FACE_MASK_PATH)) / 255.
face_mask_fix_rate = (UV_MAP_SIZE ** 2) / np.sum(face_mask_np)
foreface_ind = np.array(np.where(face_mask_np > 0)).T
if os.path.exists(UV_MEAN_SHAPE_PATH):
    mean_shape_map_np = np.load(UV_MEAN_SHAPE_PATH)
if os.path.exists(UV_EDGES_PATH):
    uv_edges = np.load(UV_EDGES_PATH)
if os.path.exists(UV_TRIANGLES_PATH):
    uv_triangles = np.load(UV_TRIANGLES_PATH)


def process_uv(uv_coordinates):
    uv_h = UV_MAP_SIZE
    uv_w = UV_MAP_SIZE
    uv_coordinates[:, 0] = uv_coordinates[:, 0] * (uv_w - 1)
    uv_coordinates[:, 1] = uv_coordinates[:, 1] * (uv_h - 1)
    uv_coordinates[:, 1] = uv_h - uv_coordinates[:, 1] - 1
    uv_coordinates = np.hstack((uv_coordinates, np.zeros((uv_coordinates.shape[0], 1))))  # add z
    return uv_coordinates


def read_uv_kpt(uv_kpt_path):
    file = open(uv_kpt_path, 'r', encoding='utf-8')
    lines = file.readlines()
    # txt is inversed
    x_line = lines[1]
    y_line = lines[0]
    uv_kpt = np.zeros((68, 2)).astype(int)
    x_tokens = x_line.strip().split(' ')
    y_tokens = y_line.strip().split(' ')
    for i in range(68):
        uv_kpt[i][0] = int(float(x_tokens[i]))
        uv_kpt[i][1] = int(float(y_tokens[i]))
    return uv_kpt


uv_coords = src.faceutil.morphable_model.load.load_uv_coords(BFM_UV_MAT_PATH)
uv_coords = process_uv(uv_coords)
uv_kpt_ind = read_uv_kpt(UV_KPT_INDEX_PATH)


def get_kpt_from_uvm(uv_map):
    # from uv map
    kpt = uv_map[uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
    return kpt


def uvm2mesh(uv_position_map, only_foreface=True, is_extra_triangle=False):
    """
    if no texture map is provided, translate the position map to a point cloud
    :param uv_position_map:
    :param uv_texture_map:
    :param only_foreface:
    :return:
    """
    uv_h, uv_w = UV_MAP_SIZE, UV_MAP_SIZE
    vertices = []
    colors = []
    triangles = []
    triangles = [[t[0] * uv_w + t[1], t[4] * uv_w + t[5], t[2] * uv_w + t[3], ] for t in uv_triangles]
    for i in range(uv_h):
        for j in range(uv_w):
            vertices.append(uv_position_map[i][j])
            colors.append([25, 25, 50, 128])
    # for i in range(uv_h):
    #     for j in range(uv_w):
    #         if not only_foreface:
    #             vertices.append(uv_position_map[i][j])
    #             colors.append(np.array([64, 64, 64]))
    #             pa = i * uv_h + j
    #             pb = i * uv_h + j + 1
    #             pc = (i - 1) * uv_h + j
    #             pd = (i + 1) * uv_h + j + 1
    #             if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
    #                 triangles.append([pa, pb, pc])
    #                 triangles.append([pa, pc, pb])
    #                 triangles.append([pa, pb, pd])
    #                 triangles.append([pa, pd, pb])
    #         else:
    #             if face_mask_np[i, j] == 0:
    #                 vertices.append(np.array([-1, -1, -1]))
    #                 colors.append(np.array([0, 0, 0]))
    #                 continue
    #             else:
    #                 vertices.append(uv_position_map[i][j])
    #                 colors.append(np.array([128, 0, 128]))
    #                 pa = i * uv_h + j
    #                 pb = i * uv_h + j + 1
    #                 pc = (i - 1) * uv_h + j
    #                 pd = (i + 1) * uv_h + j + 1
    #                 if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
    #                     if not face_mask_np[i, j + 1] == 0:
    #                         if not face_mask_np[i - 1, j] == 0:
    #                             triangles.append([pa, pb, pc])
    #                             triangles.append([pa, pc, pb])
    #                         if not face_mask_np[i + 1, j + 1] == 0:
    #                             triangles.append([pa, pb, pd])
    #                             triangles.append([pa, pd, pb])

    vertices = np.array(vertices)
    triangles = np.array(triangles)
    colors = np.array(colors)
    # verify_face = mesh.render.render_colors(verify_vertices, verify_triangles, verify_colors, height, width,
    #                                         channel)
    face_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return face_mesh


def get_uv_edges():
    """
    [x1,y1,x2,y2]
    :return:
    """
    uv_h, uv_w = UV_MAP_SIZE, UV_MAP_SIZE
    edges = []
    for i in range(uv_h):
        for j in range(uv_w):
            if face_mask_np[i, j] == 0:
                continue
            else:
                if i < uv_h - 1 and j < uv_w - 1:
                    if face_mask_np[i, j + 1] != 0:
                        edges.append([i, j, i, j + 1])
                    if face_mask_np[i + 1, j] != 0:
                        edges.append([i, j, i + 1, j + 1])

    # np.save('data/uv_data/uv_edges.npy', edges)
    return edges


def get_uv_triangles():
    """
    [x1,y1,x2,y2,x3,y3]
    :return:
    """

    uv_h, uv_w = UV_MAP_SIZE, UV_MAP_SIZE
    triangles = []
    for i in range(uv_h):
        for j in range(uv_w):
            if face_mask_np[i, j] == 0:
                continue
            else:
                if i < uv_h - 1 and j < uv_w - 1:
                    if face_mask_np[i, j + 1] != 0 and face_mask_np[i + 1, j] != 0:
                        triangles.append([i, j, i + 1, j, i, j + 1])

                if i > 0 and j < uv_w - 1:
                    if face_mask_np[i, j + 1] != 0 and face_mask_np[i - 1, j + 1] != 0:
                        triangles.append([i, j, i, j + 1, i - 1, j + 1])

    # np.save('data/uv_data/uv_triangles.npy', triangles)
    return triangles
