import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.dataset.uv_face import get_kpt_from_uvm

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def plot_verts(img, verts):
    ret_img = np.array(img.copy())
    for v in verts:
        cv2.circle(ret_img, (int(v[0]), int(v[1])), 2, (255, 0, 0))
    return ret_img


def plot_kpt(image, kpt, color_param=0):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    image = np.array(image.copy())
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (0 + color_param, 0, 255 - color_param), 2)
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (0 + color_param, 0, 255 - color_param), 1)
    return image


def compare_kpt(posmap, gtposmap, image):
    kpt1 = get_kpt_from_uvm(posmap)
    kpt2 = get_kpt_from_uvm(gtposmap)
    ploted = plot_kpt(image, kpt1)
    ploted = plot_kpt(ploted, kpt2, color_param=int(255))
    return ploted


def demo_kpt(posmap, image):
    kpt = get_kpt_from_uvm(posmap)
    ploted = plot_kpt(image, kpt)
    return ploted
