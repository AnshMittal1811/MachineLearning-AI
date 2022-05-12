import functools
import cv2
import numpy as np
import imageio
from glob import glob
import os
import shutil
import skimage
import io

def load_rgb(path, sidelength=None,shift=True):
    img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    if shift:
        img -= 0.5
        img *= 2.
    return img


def load_depth(path, sidelength=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img *= 1e-4

    if len(img.shape) == 3:
        img = img[:, :, :1]
        img = img.transpose(2, 0, 1)
    else:
        img = img[None, :, :]
    return img


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def load_numpy_hdf5(instance_ds, key):
    rgb_ds = instance_ds['rgb']
    raw = rgb_ds[key][...]
    s = raw.tostring()
    f = io.BytesIO(s)

    img = imageio.imread(f)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img -= 0.5
    img *= 2.

    return img


def load_rgb_hdf5(instance_ds, key, sidelength=None):
    rgb_ds = instance_ds['rgb']
    raw = rgb_ds[key][...]
    s = raw.tostring()
    f = io.BytesIO(s)

    img = imageio.imread(f)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.

    return img


def load_pose_hdf5(instance_ds, key):
    pose_ds = instance_ds['pose']
    raw = pose_ds[key][...]
    ba = bytearray(raw)
    s = ba.decode('ascii')

    lines = s.splitlines()

    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        # processed_pose = pose.squeeze()
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()
    #     processed_pose = np.asarray(lines).astype(np.float32).squeeze()
    #
    # pos = processed_pose[:3, 3]
    #
    # yaw = np.arctan2(processed_pose[1, 0], processed_pose[0, 0])
    # roll = np.arctan2(-processed_pose[2, 0], np.sqrt(processed_pose[2, 1] ** 2 + processed_pose[2, 2] ** 2))
    # pitch = np.arctan2(processed_pose[2, 1], processed_pose[2, 2])
    #
    # return np.array([pos[0], pos[1], pos[2], yaw, pitch])


def load_params(filename):
    lines = open(filename).read().splitlines()

    params = np.array([float(x) for x in lines[0].split()]).astype(np.float32).squeeze()
    return params

def create_seg_map(img, color_idx_map, background_idx = 0):
    # map from colors to labeled ints
    seg_map = np.ones([img.shape[0], 1])
    for idx, color in enumerate(img):
        color = tuple(color)
        if color not in color_idx_map: # revert to background label
            seg_map[idx] = background_idx
        else: seg_map[idx] = color_idx_map[color]
    return seg_map

def load_seg(path, sidelength=None, data_type="cars"):

    if data_type == "cars":
        label_color_map = {"windows": (0,0,255), "lights": (255,255,0), "background": (255,255,255),
                     "spoiler": (0,255,255), "car": (0,255,0), "tires": (255,0,0)}
    elif data_type == "rooms":
        label_color_map = {"floor": (0, 0, 255), "cone": (255, 0, 255), "background": (255, 255, 255),
                           "cylinder": (255, 0, 0), "wall": (0, 255, 0), "sphere": (0, 255, 255),
                           "cube":(255,255,0)}

    else: raise IOError("please provide a valid segmentation data type, i.e., cars or rooms")

    labels = label_color_map.keys()
    label_idx_map = {label: idx for idx, label in enumerate(labels)}
    color_idx_map = {color: (label_idx_map[label],) for label, color in label_color_map.items()} # inverted for ease of access

    img = imageio.imread(path)[:, :, :3]
    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img = img.reshape(-1, 3)
    seg_map = create_seg_map(img, color_idx_map, background_idx = label_idx_map["background"])

    return seg_map, {idx[0]: color for color, idx in color_idx_map.items()}

def load_seg_hdf5(instance_ds, key, sidelength=None, data_type="cars"):

    if data_type == "cars":
        label_color_map = {"windows": (0,0,255), "lights": (255,255,0), "background": (255,255,255),
                     "spoiler": (0,255,255), "car": (0,255,0), "tires": (255,0,0)}
    elif data_type == "rooms":
        label_color_map = {"floor": (0, 0, 255), "cone": (255, 0, 255), "background": (255, 255, 255),
                           "cylinder": (255, 0, 0), "wall": (0, 255, 0), "sphere": (0, 255, 255),
                           "cube":(255,255,0)}

    else: raise IOError("please provide a valid segmentation data type, i.e., cars or rooms")

    labels = label_color_map.keys()
    label_idx_map = {label: idx for idx, label in enumerate(labels)}
    color_idx_map = {color: (label_idx_map[label],) for label, color in label_color_map.items()} # inverted for ease of access

    sem_ds = instance_ds['labels']
    raw = sem_ds[key][...]
    s = raw.tostring()
    f = io.BytesIO(s)

    img = imageio.imread(f)[:, :, :3]
    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img = img.reshape(-1, 3)
    seg_map = create_seg_map(img, color_idx_map, background_idx = label_idx_map["background"])

    return seg_map, {idx[0]: color for color, idx in color_idx_map.items()}


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def train_val_split(object_dir, train_dir, val_dir):
    dirs = [os.path.join(object_dir, x) for x in ['pose', 'rgb', 'depth']]
    data_lists = [sorted(glob(os.path.join(dir, x)))
                  for dir, x in zip(dirs, ['*.txt', "*.png", "*.png"])]

    cond_mkdir(train_dir)
    cond_mkdir(val_dir)

    [cond_mkdir(os.path.join(train_dir, x)) for x in ['pose', 'rgb', 'depth']]
    [cond_mkdir(os.path.join(val_dir, x)) for x in ['pose', 'rgb', 'depth']]

    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(val_dir, 'intrinsics.txt'))
    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(train_dir, 'intrinsics.txt'))

    for data_name, data_ending, data_list in zip(['pose', 'rgb', 'depth'], ['.txt', '.png', '.png'], data_lists):
        val_counter = 0
        train_counter = 0
        for i, item in enumerate(data_list):
            if not i % 3:
                shutil.copy(item, os.path.join(train_dir, data_name, "%06d" % train_counter + data_ending))
                train_counter += 1
            else:
                shutil.copy(item, os.path.join(val_dir, data_name, "%06d" % val_counter + data_ending))
                val_counter += 1


def glob_ext(path,exts):
    elems = []
    for ext in exts:
        elems.extend(glob(os.path.join(path, '*'+ext)))
    return elems

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def read_view_direction_rays(direction_file):
    img = cv2.imread(direction_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img -= 40000
    img /= 10000
    return img


def transform_viewpoint(v):
    """Transforms the viewpoint vector into a consistent representation"""

    return np.concatenate([v[:, :3],
                           np.cos(v[:, 3:4]),
                           np.sin(v[:, 3:4]),
                           np.cos(v[:, 4:5]),
                           np.sin(v[:, 4:5])], 1)


def euler2mat(z=0, y=0, x=0):
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def look_at(vec_pos, vec_look_at):
    z = vec_look_at - vec_pos
    z = z / np.linalg.norm(z)

    x = np.cross(z, np.array([0., 1., 0.]))
    x = x / np.linalg.norm(x)

    y = np.cross(x, z)
    y = y / np.linalg.norm(y)

    view_mat = np.zeros((3, 3))

    view_mat[:3, 0] = x
    view_mat[:3, 1] = y
    view_mat[:3, 2] = z

    return view_mat
