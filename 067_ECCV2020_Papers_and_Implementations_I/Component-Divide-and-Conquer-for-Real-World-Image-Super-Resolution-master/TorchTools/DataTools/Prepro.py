from ..Functions.functional import *
import numpy as np
import cv2
import os
from os.path import basename
from .FileTools import _all_images

def _id(x):
    """
    return x
    :param x:
    :return:
    """
    return x


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


def _255_to_tanh(x):
    """
    range [0, 255] to range [-1, 1]
    :param x:
    :return:
    """
    return (x - 127.5) / 127.5


def _tanh_to_255(x):
    """
    range [-1. 1] to range [0, 255]
    :param x:
    :return:
    """
    return x * 127.5 + 127.5


# TODO: _sigmoid_to_255(x), _255_to_sigmoid(x)
# def _sigmoid_to_255(x):
# def _255_to_sigmoid(x):


def random_pre_process(img):
    """
    Random pre-processing the input Image
    :param img: PIL.Image
    :return: PIL.Image
    """
    if bool(random.getrandbits(1)):
        img = hflip(img)
    if bool(random.getrandbits(1)):
        img = vflip(img)
    # return img
    # angle = random.randrange(-15, 15)
    angle = random.randint(0, 3) * 90
    return rotate(img, angle)


def random_pre_process_pair(hr, lr, lr_patch_size, scale):
    """
    For Real Paired Images,
    Crop hr, lr images to patches, and random pre-processing
    :param hr, lr: PIL.Image
    :param lr_patch_size: lr patches size
    :param scale: upsample scale
    :return: PIL.Image
    """
    w, h = lr.size
    startx = random.randint(0, w - lr_patch_size)
    starty = random.randint(0, h - lr_patch_size)
    hr_patch = hr.crop((startx * scale, starty * scale,
                        (startx + lr_patch_size) * scale, (starty + lr_patch_size) * scale))
    lr_patch = lr.crop((startx, starty, startx + lr_patch_size, starty + lr_patch_size))

    if bool(random.getrandbits(1)):
        hr_patch = hflip(hr_patch)
        lr_patch = hflip(lr_patch)
    if bool(random.getrandbits(1)):
        hr_patch = vflip(hr_patch)
        lr_patch = vflip(lr_patch)
    # angle = random.randrange(-15, 15)
    angle = random.randint(0, 3) * 90
    return rotate(hr_patch, angle), rotate(lr_patch, angle)


def dataset_crop(dataroot, save_dir, crop_size=80, save_interval=3000):
    """
    crop dataset to train size for speed up training
    :param dataroot:
    :param save_dir:
    :param crop_size: train size * scala
    :return:
    """
    im_names = _all_images(dataroot)
    sum_cnt = 0

    save_path = os.path.join(save_dir, str(sum_cnt // save_interval))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for im_name in im_names:
        cnt = 0
        prefix = os.path.splitext(basename(im_name))[0]
        im = cv2.imread(im_name)
        h, w, c = im.shape
        for i in range(h // crop_size):
            for j in range(w // crop_size):
                patch = im[i * crop_size:(i + 1) * crop_size, j * crop_size: (j + 1) * crop_size, :]
                patch_name = '%s_%d.png' % (prefix, cnt)
                cnt += 1
                sum_cnt += 1

                if sum_cnt % save_interval == 0:
                    save_path = os.path.join(save_dir, str(sum_cnt // save_interval))
                    print(save_path)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

                cv2.imwrite(os.path.join(save_path, patch_name), patch)
                print('[%d] saving: %s' % (sum_cnt, patch_name))


