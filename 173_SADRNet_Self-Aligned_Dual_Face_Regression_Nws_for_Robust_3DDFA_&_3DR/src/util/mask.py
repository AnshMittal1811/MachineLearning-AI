import numpy as np
from src.dataset.uv_face import face_mask_np
from config import *
import cv2
from src.dataset.uv_face import uvm2mesh
from src.visualize.render_mesh import render_face_orthographic

face_mask_np3d = np.stack([face_mask_np, face_mask_np, face_mask_np], axis=2)


# for init image visibility
def get_image_attention_mask(posmap, mode='hard'):
    """
    需要加一个正态分布吗？
    """
    p = (posmap * face_mask_np3d).clip(0, 255).astype(int)
    mask = np.zeros((UV_MAP_SIZE, UV_MAP_SIZE))
    # for i in range(height):
    #     for j in range(width):
    #         [x, y, z] = posmap[i, j]
    #         x = int(x)
    #         y = int(y)
    #         mask[y, x] = 1
    mask[p[:, :, 1], p[:, :, 0]] = 1

    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = np.ceil(np.array(blur)).astype(np.uint8) * 255
    return mask


def get_image_attention_mask_v2(uv_map):
    face_mesh = uvm2mesh(uv_map / UV_MAP_SIZE * 2)
    ret = np.array(render_face_orthographic(face_mesh).astype(np.float32))
    ret[ret == 255] = 0
    ret[ret != 0] = 1
    blurred = cv2.GaussianBlur(ret, (3, 3), 0)
    mask = np.ceil(np.array(blurred)).astype(np.float32)
    return (mask * 255).astype(np.uint8)
