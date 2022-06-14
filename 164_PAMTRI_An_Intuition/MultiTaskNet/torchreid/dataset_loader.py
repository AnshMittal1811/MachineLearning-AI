# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License-NC
# See LICENSE.txt for details
#
# Author: Zheng Tang (tangzhengthomas@gmail.com)


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import numpy as np
import os.path as osp
import io

import torch
from torch.utils.data import Dataset


def read_image_color(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_image_grayscale(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    """Image ReID Dataset"""
    def __init__(self, dataset, keyptaware=True, heatmapaware=True, segmentaware=True, transform=None, imagesize=None):
        self.dataset = dataset
        self.keyptaware = keyptaware
        self.heatmapaware = heatmapaware
        self.segmentaware = segmentaware
        self.transform = transform
        self.imagesize = imagesize

        self.segments = [(5, 15, 16, 17), (5, 6, 12, 15), (6, 10, 11, 12), 
                         (23, 33, 34, 35), (23, 24, 30, 33), (24, 28, 29, 30), 
                         (10, 11, 29, 28), (11, 12, 30, 29), (12, 13, 31, 30), 
                         (13, 14, 32, 31), (14, 15, 33, 32), (15, 16, 34, 33), 
                         (16, 17, 35, 34)]
        self.conf_thld = 0.5

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_chnls = []

        img_path, vid, camid, vcolor, vtype, vkeypt, heatmap_dir_path, segment_dir_path = self.dataset[index]
        img_orig = read_image_color(img_path)
        height_orig, width_orig, channels = img_orig.shape
        img_b, img_g, img_r = cv2.split(img_orig)
        img_chnls.extend([img_r, img_g, img_b])

        if self.heatmapaware:
            for h in range(36):
                heatmap_path = osp.join(heatmap_dir_path, "%02d.jpg" % h)
                heatmap = read_image_grayscale(heatmap_path)
                heatmap = cv2.resize(heatmap, dsize=(width_orig, height_orig))
                img_chnls.append(heatmap)

        if self.segmentaware:
            for s in range(len(self.segments)):
                segment_flag = True
                for k in self.segments[s]:
                    if vkeypt[k*3+2] < self.conf_thld:
                        segment_flag = False
                        break
                if segment_flag:
                    segment_path = osp.join(segment_dir_path, "%02d.jpg" % s)
                    segment = read_image_grayscale(segment_path)
                    segment = cv2.resize(segment, dsize=(width_orig, height_orig))
                else:
                    segment = np.zeros((height_orig, width_orig), np.uint8)
                img_chnls.append(segment)

        assert self.transform is not None
        img = np.stack(img_chnls, axis=2)
        img = self.transform(img, vkeypt)
        vkeypt = np.asarray(vkeypt)

        # normalize keypt
        if self.keyptaware:
            for k in range(vkeypt.size):
                if k % 3 == 0:
                    vkeypt[k] = (vkeypt[k] / float(self.imagesize[0])) - 0.5
                elif k % 3 == 1:
                    vkeypt[k] = (vkeypt[k] / float(self.imagesize[1])) - 0.5
                elif k % 3 == 2:
                    vkeypt[k] -= 0.5

        return img, vid, camid, vcolor, vtype, vkeypt
