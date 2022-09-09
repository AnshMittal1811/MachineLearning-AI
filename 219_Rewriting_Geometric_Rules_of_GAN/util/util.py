import os
import importlib
import argparse
from argparse import Namespace
from collections import OrderedDict

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lib.mls import mls_affine_deformation


def load_keypoints(path, w, h):
    fixed_keypts = [[0, 0, 0, 0], [0, h - 1, 0, h - 1], [w - 1, 0, w - 1, 0], [w - 1, h - 1, w - 1, h - 1]]
    keypoints = np.load(path)
    return np.concatenate([keypoints, fixed_keypts], axis=0)


def load_mask(path, size=None, interp=Image.BILINEAR):
    """Load mask stored as PNG into a numpy array."""
    mask = Image.open(path).convert('L')
    if size and size != mask.size:
        mask = mask.resize(size, interp)
    mask = np.asarray(mask) / 255
    return mask.astype(np.float32)


def get_warp_grid(size, keypoints):
    # getting a (x, y) meshgrid
    height, width = size
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vx, vy = np.meshgrid(gridX, gridY)

    # parsing the keypoints
    keypoints = np.floor(np.array(keypoints, dtype=np.float32))
    src = keypoints[:, :2]
    tgt = keypoints[:, 2:]

    # getting the deformation grid, singularities on the control point need extra care
    deform = mls_affine_deformation(vx, vy, src, tgt, alpha=1)
    deform = np.array(deform)
    idx_i, idx_j = tgt[:, 1].astype(int), tgt[:, 0].astype(int)
    deform[:, idx_i, idx_j] = src[:, ::-1].T

    # mls_rigd_deformation output (y, x) grid, want to return (x, y) grid instead
    return deform[::-1, :, :]


def get_warp_grid_normalized(keypoints, image_res, target_res):
    # get warping grid from keypoints, the range is in pixel length
    vgrid = get_warp_grid(image_res, keypoints)

    # normalize grid range to [-1, 1]
    h, w = image_res
    vgrid[0, :, :] = 2.0 * vgrid[0, :, :] / max(w - 1, 1) - 1.0
    vgrid[1, :, :] = 2.0 * vgrid[1, :, :] / max(h - 1, 1) - 1.0

    # Transpose to [x, y, flow] and resize the flow to target resolution
    vgrid = vgrid.transpose((1, 2, 0))
    vgrid = cv2.resize(vgrid, target_res[::-1], interpolation=cv2.INTER_LANCZOS4)

    deform = np.copy(vgrid)
    return deform


def warp_by_keypoints(image, keypoints, interp=cv2.INTER_CUBIC):
    is_pil = type(image) == Image.Image
    if is_pil:
        image = np.asarray(image)

    deform = get_warp_grid(image.shape[:2], keypoints)
    dx, dy = deform[0], deform[1]
    warped = cv2.remap(image, dx, dy, interp, borderMode=cv2.BORDER_REFLECT)
    if is_pil:
        warped = Image.fromarray(warped)

    return warped


def visualize_features(feats):
    feats = feats.detach().cpu()
    processed = []
    for feat in feats:
        for channel in feat:
            min_val, max_val = torch.min(channel), torch.max(channel)
            channel = (channel - min_val) / (max_val - min_val)
            channel = channel * 2 - 1
            processed.append(channel)
    return torch.stack(processed)[:, None, :, :]


def visualize_features_magnitude(feats):
    feats = feats.detach().cpu()
    processed = []
    for feat in feats:
        feat = torch.norm(feat, dim=0)
        min_val, max_val = torch.min(feat), torch.max(feat)
        feat = (feat - min_val) / (max_val - min_val)
        feat = feat * 2 - 1
        processed.append(feat)
    return torch.stack(processed)[:, None, :, :]


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_spatialmap(spatial_map):
    """Plot a 2D numpy array of spatial map and outputs a PIL Image."""
    fig = plt.figure()
    plt.imshow(spatial_map)
    plt.colorbar()
    map_pil = fig2img(fig)
    plt.close()
    return map_pil


def extract_size(layer):
    """According to the layer name, extract the resolution of the output features."""
    # stylegan2
    if '.b' in layer:
        sz = layer.replace('synthesis.b', '')
        sz = int(sz.split('.')[0])
    # stylegan3
    else:
        sz = layer.replace('synthesis.', '')
        sz = int(sz.split('_')[1])
    return sz


def slice_ordered_dict(d, start, end):
    assert type(d) == OrderedDict, f"d must be an OrderedDict, but get type {type(d)}"
    return OrderedDict(list(d.items())[start:end])


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def clip_image_size(img, max_size=1024):
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    mult = 1024 / max(h, w)
    new_h, new_w = int(h * mult), int(w * mult)
    resized = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(resized)


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=2):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if len(image_tensor.shape) == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.shape[0]):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile is not False:
            tile = max(min(images_np.shape[0] // 2, 4), 1) if tile is True else tile
            images_tiled = tile_images(images_np, picturesPerRow=tile)
            return images_tiled
        else:
            return images_np

    if len(image_tensor.shape) == 2:
        assert False
    image_numpy = image_tensor.detach().cpu().numpy() if type(image_tensor) is not np.ndarray else image_tensor
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, axis=2)
    return image_numpy.astype(imtype)


def toPILImage(images, tile=None):
    if isinstance(images, list):
        if all(['tensor' in str(type(image)).lower() for image in images]):
            return toPILImage(torch.cat([im.cpu() for im in images], dim=0), tile)
        return [toPILImage(image, tile=tile) for image in images]

    if 'ndarray' in str(type(images)).lower():
        return toPILImage(torch.from_numpy(images))

    assert 'tensor' in str(type(images)).lower(), "input of type %s cannot be handled." % str(type(images))

    if tile is None:
        max_width = 2560
        tile = min(images.size(0), int(max_width / images.size(3)))

    return Image.fromarray(tensor2im(images, tile=tile))


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
