import sys
sys.path.append('../core')

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from frame_utils import *
# from utils import *
# from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import math
import json
import matplotlib.pyplot as plt
from fusion import *
# from sklearn.neighbors import KDTree
# import open3d as o3d
import pickle
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
parser.add_argument('--scan', default=None, type=str)
parser.add_argument('--folder1', type=str)
parser.add_argument('--folder2', type=str)
parser.add_argument('--th', type=float, default=0.01)
args = parser.parse_args()


folder = args.folder

folder1 = args.folder1
folder2 = args.folder2
th = args.th

if args.scan is not None:
    scans = [args.scan, ]
else:
    scans_1 = [f.name for f in os.scandir(folder1) if f.is_dir()]
    scans_2 = [f.name for f in os.scandir(folder2) if f.is_dir()]
    assert (scans_1==scans_2), "scans mismatch in folder1 and folder2"
    scans = scans_1

# subprocess.call(f"mkdir -p {folder}/choices", shell=True)

for scan in scans:
    names = [f for f in os.listdir(os.path.join(folder1, f"{scan}/depths")) if f.endswith('.pfm')]
    n = len(names)

    subprocess.call(f"mkdir -p {folder}/{scan}/depths", shell=True)


    for i in range(n):
        # if i % 10: continue
        im1 = readPFM(os.path.join(folder1, f"{scan}/depths/{names[i]}"))
        im2 = readPFM(os.path.join(folder2, f"{scan}/depths/{names[i]}"))

        im1 = cv2.resize(im1, im2.shape[::-1], interpolation=cv2.INTER_NEAREST)

        mask = np.abs(im1 - im2) < th * im1
        mask = np.logical_and(mask, im1 > 0.0)
        # if i % 10 == 0: cv2.imwrite(f"{folder}/choices/{scan}_{names[i][:-4]}.png", np.array(mask, dtype=np.uint8) * 255)
        # continue

        # im1 = np.where(im1 == 0, 0, 1 / im1)
        # im2 = np.where(im2 == 0, 0, 1 / im2)

        im = np.where(mask, im2, im1)

        # plt.subplot(1, 3, 1)
        # plt.imshow(im1)
        # plt.subplot(1, 3, 2)
        # plt.imshow(im2)
        # plt.subplot(1, 3, 3)
        # plt.imshow(mask)
        # plt.savefig("test.png")
        max_depth = np.max(im)
        depth_to_vis = (im / max_depth * 255.0).astype(np.uint8)
        name_base = 'depth_visual_' + os.path.splitext(names[i])[0] + '.png'
        cv2.imwrite(f"{folder}/{scan}/depths/{name_base}", depth_to_vis)

        write_pfm(f"{folder}/{scan}/depths/{names[i]}", im)
        print(i, scan)
