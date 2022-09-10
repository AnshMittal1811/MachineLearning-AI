from meters import Meters
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from train import train
from transform import transform
from save import save_pic, save_params
from load import load
import torch
import numpy as np
from easydict import EasyDict as edict
import time
import sys
from glob import glob
import os
import argparse
import json

sys.path.append(os.getcwd())

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Fit SMPL')
    parser.add_argument('--exp', dest='exp',
                        help='Define exp name',
                        default=time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())), type=str)
    parser.add_argument('--dataset_name', dest='dataset_name',
                        help='select dataset',
                        default='', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='path of dataset',
                        default=None, type=str)
    args = parser.parse_args()
    return args


def get_config(args):
    config_path = 'configs/{}.json'.format(args.dataset_name)
    with open(config_path, 'r') as f:
        data = json.load(f)
    cfg = edict(data.copy())
    if not args.dataset_path == None:
        cfg.DATASET.PATH = args.dataset_path
    return cfg


def set_device(USE_GPU):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.join(os.getcwd(), 'exp', args.exp)
    assert not os.path.exists(cur_path), 'Duplicate exp name'
    os.mkdir(cur_path)

    cfg = get_config(args)
    json.dump(dict(cfg), open(os.path.join(cur_path, 'config.json'), 'w'))

    print("Start print log")

    device = set_device(USE_GPU=cfg.USE_GPU)
    print('using device: {}'.format(device))

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=cfg.MODEL.GENDER,
        model_root='smplpytorch/native/models')

    meters = Meters()
    file_num = 0

    paths = glob(cfg.DATASET.PATH + "/*/*.npy")
    for path_full in paths:

        file_num += 1
        print(
            'Processing file: {}    [{} / {}]'.format(path_full, file_num, len(paths)))
        target = torch.from_numpy(
            transform(args.dataset_name, load(args.dataset_name, path_full))).float()
        print("target shape:{}".format(target.shape))
        res = train(smpl_layer, target, device, args, cfg, meters)
        meters.update_avg(meters.min_loss, k=target.shape[0])
        meters.reset_early_stop()
        print("avg_loss:{:.4f}".format(meters.avg))

        save_params(res, path_full, cfg.DATASET.TARGET_PATH)
        # save_pic(res,smpl_layer,file,logger,args.dataset_name,target)
        torch.cuda.empty_cache()
    print("Fitting finished! Average loss:     {:.9f}".format(meters.avg))
