import argparse
import os
import sys
import time
from math import log10

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def _video_image_file(path):
    """
    Data Store Format:

    Data Folder
        |
        |-Video Folder
        |   |-Video Frames (images)
        |
        |-Video Folder
            |-Video Frames (images)

        ...

        |-Video Folder
            |-Video Frames (images)

    :param path: path to Data Folder, absolute path
    :return: 2D list of str, the path is absolute path
            [[Video Frames], [Video Frames], ... , [Video Frames]]
    """
    abs_path = os.path.abspath(path)
    video_list = os.listdir(abs_path)
    video_list.sort()
    frame_list = [None] * len(video_list)
    for i in range(len(video_list)):
        video_list[i] = os.path.join(path, video_list[i])
        frame_list[i] = os.listdir(video_list[i])
        for j in range(len(os.listdir(video_list[i]))):
            frame_list[i][j] = os.path.join(video_list[i], frame_list[i][j])
        frame_list[i].sort()
    return frame_list


def video_frame_names(path):
    video, frame = os.path.split(path)
    _, video = os.path.split(video)
    return video, frame


def sample_info_video(video_frames, time_window, time_stride):
    samples = [0] * len(video_frames)
    area_sum_samples = [0] * len(video_frames)
    for i, video in enumerate(video_frames):
        samples[i] = (len(video) - time_window) // time_stride
        if i != 0:
            area_sum_samples[i] = sum(samples[:i])
    return samples, area_sum_samples


def _sample_from_videos_frames(path, time_window, time_stride):
    """
    Sample from video frames files
    :param path: path to Data Folder, absolute path
    :param time_window: number of frames in one sample
    :param time_stride: strides when sample frames
    :return: 2D list of str, absolute path to each frames
            [[Sample Frames], [Sample Frames], ... , [Sample Frames]]
    """
    video_frame_list = _video_image_file(path)
    sample_list = list()
    for video in video_frame_list:
        assert isinstance(video, list), "Plz check video_frame_list = _video_image_file(path) should be 2D list"
        for i in range(0, len(video), time_stride):
            sample = video[i:i + time_window]
            if len(sample) != time_window:
                break
            sample.append(video[i + (time_window // 2)])
            sample_list.append(sample)
    return sample_list


# TODO: large sample number function
def _sample_from_videos_frames_large(path, time_window, time_stride):
    """
    write to a file, return one sample once. use pointer
    :param path:
    :param time_window:
    :param time_stride:
    :return:
    """
    pass


def _image_file(path):  # TODO: wrong function
    """
    return list of images in the path
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    abs_path = os.path.abspath(path)
    image_files = os.listdir(abs_path)
    for i in range(len(image_files)):
        if (not os.path.isdir(image_files[i])) and (_is_image_file(image_files[i])):
            image_files[i] = os.path.join(abs_path, image_files[i])
    return image_files


def _all_images(path, sort=True):
    """
    return all images in the folder
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    # TODO: Tail Call Elimination
    abs_path = os.path.abspath(path)
    image_files = list()
    for subpath in os.listdir(abs_path):
        if os.path.isdir(os.path.join(abs_path, subpath)):
            image_files = image_files + _all_images(os.path.join(abs_path, subpath))
        else:
            if _is_image_file(subpath):
                image_files.append(os.path.join(abs_path, subpath))
    if sort:
        image_files.sort()
    return image_files



def stride_crop_folder(src_folder, dst_folder, patch_size, stride, filter_func=None):
    """
    Stride crop Image to patches
    :param src_folder:  large image folder
    :param dst_folder:  patches folder
    :param patch_size:
    :param stride:
    :param filter_func: filter function, like use var or mean to filter patches
    :return: None
    """
    im_paths = _all_images(src_folder)
    for im_path in im_paths:
        im_name = os.path.basename(im_path).split('.')[0]
        im = Image.open(im_path)
        w, h = im.size
        cnt = 0
        for x in range((w - patch_size) // stride + 1):
            for y in range((h - patch_size) // stride + 1):
                startx = x * stride
                starty = y * stride
                patch = im.crop([startx, starty, startx + patch_size, starty + patch_size])

                if (filter_func is not None) and (not filter_func(patch)):
                    cnt += 1
                    continue

                patch_name = '%s_%d.png' % (im_name, cnt)
                cnt += 1
                patch.save(os.path.join(dst_folder, patch_name))
                print('saving: %s' % os.path.join(dst_folder, patch_name))
