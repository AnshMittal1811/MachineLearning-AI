import os

import numpy as np
import skimage.io as io
from scipy import misc

import torch
from torch.autograd import Variable

from ..DataTools.FileTools import _image_file
from ..DataTools.Loaders import pil_loader, load_to_tensor


RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6
ncols = sum([RY, YG, GC, CB, BM, MR])


LEFT_EYE = [36, 37, 38, 39, 40, 41]
LEFT_EYEBROW = [17, 18, 19, 20, 21]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
RIGHT_EYEBROW = [22, 23, 24, 25, 26]
MOUTH = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
LEFT_MOUTH = [48, 60]
RIGHT_MOUTH = [54, 64]
LEFT_MOST = [0, 1, 2]
RIGHT_MOST = [14, 15, 16]
TOP_MOST = [18, 19, 20, 23, 24, 25]
DOWN_MOST = [7, 8, 9]
NOSE_TIP = [31, 32, 33, 34, 35]


def _id(x):
    """
    return x
    :param x:
    :return:
    """
    return x


def _add_batch_one(tensor):
    """
    Return a tensor with size (1, ) + tensor.size
    :param tensor: 2D or 3D tensor
    :return: 3D or 4D tensor
    """
    return tensor.view((1, ) + tensor.size())


def _remove_batch(tensor):
    """
    Return a tensor with size tensor.size()[1:]
    :param tensor: 3D or 4D tensor
    :return: 2D or 3D tensor
    """
    return tensor.view(tensor.size()[1:])


def _sigmoid_to_tanh(x):
    """
    range [0, 1] to range [-1, 1]
    :param x: tensor type
    :return: tensor
    """
    return (x - 0.5) * 2.


def _tanh_to_sigmoid(x):
    """
    range [-1, 1] to range [0, 1]
    :param x:
    :return:
    """
    return x * 0.5 + 0.5


def test_pool(path, cuda=True, mode='Y', normalization_func=_sigmoid_to_tanh):
    img_path_list = _image_file(path)
    test_img_pool = list()
    for i in range(len(img_path_list)):
        pic = Variable(_add_batch_one(normalization_func(load_to_tensor(img_path_list[i], mode=mode))))
        pic.volatile = True
        if cuda:
            test_img_pool.append(pic.cuda())
        else:
            test_img_pool.append(pic)
    return test_img_pool


def high_point_to_low_point(point_set, size_h, size_l):
    """
    The three parameters will be (w, h)
    :param point: the point location
    :param size_h: high resolution image size
    :param size_l: low resolution image size
    :return:
    """
    h_1, h_2 = size_h
    l_1, l_2 = size_l
    lr_points = list()
    for point in point_set:
        x, y = point
        lr_points.append([int(round(x * (l_1 / h_1))), int(round(y * (l_2 / h_2)))])
    return np.array(lr_points)


def _center_point(point_1, point_2):
    return (np.array(point_1, dtype=np.float32) + np.array(point_2, dtype=np.float32)) // 2


def _euclid_distance(point_1, point_2):
    return np.sqrt(np.sum((point_1 - point_2) ** 2))


def _angle_point(center, point_1, point_2):
    a = _euclid_distance(point_1, point_2)
    b = _euclid_distance(center, point_2)
    c = _euclid_distance(center, point_1)
    return np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))


def _rotate_affine_matrix(center, theta, high):
    """
    :param center: rotate center
    :param theta: clock wise, rad
    :return:
    """
    x, y = center
    # y = high - y_p
    sin = np.sin(theta)
    cos = np.cos(theta)
    matrix = np.array(
        [[cos, -sin, x - x * cos + y * sin],
         [sin, cos, y - x * sin - y * cos]]
    )
    return torch.from_numpy(matrix)


def get_landmarks(img, detector, predictor):
    """
    Return landmark martix
    :param img: img read by skimage.io.imread
    :param detector: dlib.get_frontal_face_detector() instance
    :param predictor: dlib.shape_predictor('..?./shape_predictor_68_face_landmarks.dat')
    :return: landmark matrix
    """
    rects = detector(img, 1)
    return np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def _centroid(landmarks, point_list):
    """
    Return the centroid of given points
    :param point_list: point list
    :return: array(centroid) (x, y)
    """
    x = np.zeros((len(point_list),))
    y = np.zeros((len(point_list),))
    for i, p in enumerate(point_list):
        x[i] = landmarks[p][0]
        y[i] = landmarks[p][1]
    x_mean = int(x.mean())
    y_mean = int(y.mean())
    return np.array([x_mean, y_mean])


def make_color_wheel():
    """A color wheel or color circle is an abstract illustrative
       organization of color hues around a circle.
       This is for making output image easy to distinguish every
       part.
    """
    # These are chosen based on perceptual similarity
    # e.g. one can distinguish more shades between red and yellow
    #      than between yellow and green

    if ncols > 60:
        exit(1)

    color_wheel = np.zeros((ncols, 3))
    i = 0
    # RY: (255, 255*i/RY, 0)
    color_wheel[i: i + RY, 0] = 255
    color_wheel[i: i + RY, 1] = np.arange(RY) * 255 / RY
    i += RY
    # YG: (255-255*i/YG, 255, 0)
    color_wheel[i: i + YG, 0] = 255 - np.arange(YG) * 255 / YG
    color_wheel[i: i + YG, 1] = 255
    i += YG
    # GC: (0, 255, 255*i/GC)
    color_wheel[i: i + GC, 1] = 255
    color_wheel[i: i + GC, 2] = np.arange(GC) * 255 / GC
    i += GC
    # CB: (0, 255-255*i/CB, 255)
    color_wheel[i: i + CB, 1] = 255 - np.arange(CB) * 255 / CB
    color_wheel[i: i + CB, 2] = 255
    i += CB
    # BM: (255*i/BM, 0, 255)
    color_wheel[i: i + BM, 0] = np.arange(BM) * 255 / BM
    color_wheel[i: i + BM, 2] = 255
    i += BM
    # MR: (255, 0, 255-255*i/MR)
    color_wheel[i: i + MR, 0] = 255
    color_wheel[i: i + MR, 2] = 255 - np.arange(MR) * 255 / MR

    return color_wheel


def mapping_to_indices(coords):
    """numpy advanced indexing is like x[<indices on axis 0>, <indices on axis 1>, ...]
        this function convert coords of shape (h, w, 2) to advanced indices

    # Arguments
        coords: shape of (h, w)
    # Returns
        indices: [<indices on axis 0>, <indices on axis 1>, ...]
    """
    h, w = coords.shape[:2]
    indices_axis_2 = list(np.tile(coords[:, :, 0].reshape(-1), 2))
    indices_axis_3 = list(np.tile(coords[:, :, 1].reshape(-1), 1))
    return [indices_axis_2, indices_axis_3]


def flow_to_color(flow, normalized=True):
    """
    # Arguments
        flow: (h, w, 2) flow[u, v] is (y_offset, x_offset)
        normalized: if is True, element in flow is between -1 and 1, which
                    present to
    """
    color_wheel = make_color_wheel()  # (55, 3)
    h, w = flow.shape[:2]
    rad = np.sum(flow ** 2, axis=2) ** 0.5  # shape: (h, w)
    rad = np.concatenate([rad.reshape(h, w, 1)] * 3, axis=-1)
    a = np.arctan2(-flow[:, :, 1], -flow[:, :, 0]) / np.pi  # shape: (h, w) range: (-1, 1)
    fk = (a + 1.0) / 2.0 * (ncols - 1)  # -1~1 mapped to 1~ncols
    k0 = np.floor(fk).astype(np.int)
    k1 = (k0 + 1) % ncols
    f = (fk - k0).reshape((-1, 1))
    f = np.concatenate([f] * 3, axis=1)
    color0 = color_wheel[list(k0.reshape(-1))] / 255.0
    color1 = color_wheel[list(k1.reshape(-1))] / 255.0
    res = (1 - f) * color0 + f * color1
    res = np.reshape(res, (h, w, 3))  # flatten to h*w
    mask = rad <= 1
    res[mask] = (1 - rad * (1 - res))[mask]  # increase saturation with radius
    res[~mask] *= .75  # out of range

    return res
