from armatures import *
from models import *
import numpy as np
import pickle
import glob
import os
import json
from easydict import EasyDict as edict
import argparse


def get_config(args):
    config_path = 'configs/{}.json'.format(args.dataset_name)
    with open(config_path, 'r') as f:
        data = json.load(f)
    cfg = edict(data.copy())
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit SMPL')
    parser.add_argument('--dataset_name', dest='dataset_name',
                        help='select dataset',
                        default='', type=str)
    args = parser.parse_args()
    cfg = get_config(args)
    data_map = {tp[0]:t[[1]] for tp in cfg.DATASET.DATA_MAP}

    paths = glob(cfg.DATASET.PATH  + "/*/*.npy")
    for path_full in paths:
        print(path_full)
        path = os.path.normpath(path_full)
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]
        pkl_name = os.path.join(cfg.DATASET.PATH, gt_file_name, f"{full_file_name}_smpl_params.pkl")
        poses = np.load(path, allow_pickle=True)
        with open(pkl_name, "rb") as f:
            param = pickle.load(f)
      
        pose_pca = np.array(param['pose_params'][0])[3:]
        shape =  np.array(param['shape_params'][0])
        pose_glb = np.zeros([1, 3])
        mesh = KinematicModel("/itet-stor/liuzhi/net_scratch/smpl_models/smpl_neutral.pkl", SMPLArmature, scale=1)
        _, keypoints = mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb)
        # compute offset
        target_id = 12 # use neck
        source_id = data_map[target_id]
        offset = poses[source_id] - keypoints[target_id]
        outpath = os.path.join(cfg.DATASET.PATH, gt_file_name, f"{full_file_name}_smpl_model.obj")
        mesh.save_obj(outpath, offset)
