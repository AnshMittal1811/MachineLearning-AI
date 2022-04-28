"""
loads calibrations
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import os, sys
sys.path.append("/")
import json
from os.path import join, basename, dirname
import numpy as np
from data.kinect_calib import KinectCalib


def rotate_yaxis(R, t):
    "rotate the transformation matrix around z-axis by 180 degree ==>> let y-axis point up"
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    global_trans = np.eye(4)
    global_trans[0, 0] = global_trans[1, 1] = -1  # rotate around z-axis by 180
    rotated = np.matmul(global_trans, transform)
    return rotated[:3, :3], rotated[:3, 3]


def load_intrinsics(intrinsic_folder, kids):
    """
    kids: list of kinect id that should be loaded
    """
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]

    return kinects


def load_kinect_poses(config_folder, kids):
    pose_calibs = [json.load(open(join(config_folder, f"{x}/config.json"))) for x in kids]
    rotations = [np.array(pose_calibs[x]['rotation']).reshape((3, 3)) for x in kids]
    translations = [np.array(pose_calibs[x]['translation']) for x in kids]
    return rotations, translations


def load_kinects(intrinsic_folder, config_folder, kids):
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    pose_files = [join(config_folder, f"{x}/config.json") for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]
    return kinects


def load_kinect_poses_back(config_folder, kids, rotate=False):
    """
    backward transform
    rotate: kinect y-axis pointing down, if rotate, then return a transform that make y-axis pointing up
    """
    rotations, translations = load_kinect_poses(config_folder, kids)
    rotations_back = []
    translations_back = []
    for r, t in zip(rotations, translations):
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t

        trans_back = np.linalg.inv(trans) # now the y-axis point down

        r_back = trans_back[:3, :3]
        t_back = trans_back[:3, 3]
        if rotate:
            r_back, t_back = rotate_yaxis(r_back, t_back)

        rotations_back.append(r_back)
        translations_back.append(t_back)
    return rotations_back, translations_back


def availabe_kindata(input_video, kinect_count=3):
    # all available kinect videos in this folder, return the list of kinect id, and str representation
    fname_split = os.path.basename(input_video).split('.')
    idx = int(fname_split[1])
    kids = []
    comb = ''
    for k in range(kinect_count):
        file = input_video.replace(f'.{idx}.', f'.{k}.')
        if os.path.exists(file):
            kids.append(k)
            comb = comb + str(k)
        else:
            print("Warning: {} does not exist in this folder!".format(file))
    return kids, comb
