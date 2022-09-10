import numpy as np
import os
import numpy as np
from glob import glob
import argparse
import config.config_loader as cfg_loader


def read_pose3d(path, joint_num=25):
    landmarks3d_list = []
    with open(path) as f:
        landmarks3d_txt = f.read().split('\n')
        for i in range(joint_num):
            landmarks3d = landmarks3d_txt[i].split(' ')
            landmarks3d_list.append(float(landmarks3d[1]))
            landmarks3d_list.append(float(landmarks3d[2]))
            landmarks3d_list.append(float(landmarks3d[3]))
    return np.array(landmarks3d_list).reshape([joint_num, 3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    type_mode = 'test'
    paths = glob(os.path.join(
        cfg['data_path'], type_mode, '*/landmarks3d.txt'))


    for path in paths:
        np_pose = read_pose3d(path)
        path_dir = os.path.dirname(path)
        gt_file_name = path_dir.split(os.sep)[-1]
        file_type = path_dir.split(os.sep)[-2]

        out_file = os.path.join(
            cfg['data_path'], file_type + '_smpl', gt_file_name, gt_file_name + '_pose')

        if os.path.exists(out_file):
            continue
        out_dir = os.path.join(cfg['data_path'], file_type + '_smpl',
                               gt_file_name)
        os.makedirs(out_dir, exist_ok=True)
        print(out_file)
        np.save(out_file, np_pose)
