import pickle
import numpy as np
import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Detect cross joints')
    parser.add_argument('--dataset_name', dest='dataset_name',
                        help='select dataset',
                        default='', type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='path of output',
                        default=None, type=str)
    args = parser.parse_args()
    return args


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_Jtr(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    Jtr = np.array(data["Jtr"])
    return Jtr


def has_cross(joints: np.ndarray):
    return (joints[1][0]-joints[2][0]) * (joints[10][0]-joints[11][0]) < 0 or\
           (joints[13][0]-joints[14][0]) * (joints[22][0]-joints[23][0]) < 0


def cross_frames(Jtr: np.ndarray):
    ans = []
    for frame in range(Jtr.shape[0]):
        if has_cross(Jtr[frame]):
            ans.append(frame)
    return ans


def cross_detector(dir_path):
    ans = {}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(dir_path, file)
            Jtr = load_Jtr(file_path)
            ans[file] = cross_frames(Jtr)
    return ans


if __name__ == "__main__":
    args = parse_args()
    d = cross_detector(args.output_path)
    json.dump(
        d, open("./fit/output/cross_detection/{}.json"
                .format(args.dataset_name), 'w'))