"""
Testing process.

Usage:
# For KITTI Depth Completion
>> python test.py --model_cfg exp/test/test_options.py --model_path exp/test/ckpt/\[ep-00\]giter-0.ckpt \
                  --dataset kitti2017 --rgb_dir ./data/kitti2017/rgb --depth_dir ./data/kitti2015/depth
# For KITTI Stereo
>> python test.py --model_cfg exp/test/test_options.py --model_path exp/test/ckpt/\[ep-00\]giter-0.ckpt \
                  --dataset kitti2015 --root_dir ./data/kitti_stereo/data_scene_flow
"""

import os
import sys
import time
import argparse
import importlib
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from misc import utils
from misc import metric
from dataset.dataset_kitti2017 import DatasetKITTI2017
from dataset.dataset_kitti2015 import DatasetKITTI2015


DISP_METRIC_FIELD = ['err_3px', 'err_2px', 'err_1px', 'rmse', 'mae']
DEPTH_METRIC_FIELD = ['rmse', 'mae', 'mre', 'irmse', 'imae']

SEED = 100
random.seed(SEED)
np.random.seed(seed=SEED)
cudnn.deterministic = True
cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def parse_arg():
    parser = argparse.ArgumentParser(description='Sparse-Depth-Stereo testing')
    parser.add_argument('--model_cfg', dest='model_cfg', type=str, default=None,
                        help='Configuration file (options.py) of the trained model.')
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Path to weight of the trained model.')
    parser.add_argument('--dataset', dest='dataset', type=str, default='kitti2017',
                        help='Dataset used: kitti2015 / kitti2017')
    parser.add_argument('--rgb_dir', dest='rgb_dir', type=str, default='./data/kitti2017/rgb',
                        help='Directory of RGB data for kitti2017.')
    parser.add_argument('--depth_dir', dest='depth_dir', type=str, default='./data/kitti2017/depth',
                        help='Directory of depth data for kitti2015.')
    parser.add_argument('--root_dir', dest='root_dir', type=str, default='./data/kitti_stereo/data_scene_flow',
                        help='Root directory for kitti2015')
    parser.add_argument('--random_sampling', dest='random_sampling', type=float, default=None,
                        help='Perform random sampling on ground truth to obtain sparse disparity map; Only used in kitti2015')
    parser.add_argument('--no_cuda', dest='no_cuda', action='store_true',
                        help='Don\'t use gpu')
    parser.set_defaults(no_cuda=False)
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arg()

    # Import configuration file
    sys.path.append('/'.join((args.model_cfg).split('/')[:-1]))
    options = importlib.import_module(((args.model_cfg).split('/')[-1]).split('.')[0])
    cfg = options.get_config()

    # Define model and load
    model = options.get_model(cfg.model_name)
    if not args.no_cuda:
        model = model.cuda()
    train_ep, train_step = utils.load_checkpoint(model, None, None, args.model_path, True)

    # Define testing dataset (NOTE: currently using validation set)
    if args.dataset == 'kitti2017':
        dataset = DatasetKITTI2017(args.rgb_dir, args.depth_dir, 'my_test',
                                   (256, 1216), to_disparity=cfg.to_disparity, # NOTE: set image size to 256x1216
                                   fix_random_seed=True)
    elif args.dataset == 'kitti2015':
        dataset = DatasetKITTI2015(args.root_dir, 'training', (352, 1216), # NOTE: set image size to 352x1216
                                   args.random_sampling, fix_random_seed=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                        num_workers=4)

    # Perform testing
    model.eval()
    pbar = tqdm(loader)
    pbar.set_description('Testing')
    disp_meters = metric.Metrics(DISP_METRIC_FIELD)
    disp_avg_meters = metric.MovingAverageEstimator(DISP_METRIC_FIELD)
    depth_meters = metric.Metrics(DEPTH_METRIC_FIELD)
    depth_avg_meters = metric.MovingAverageEstimator(DEPTH_METRIC_FIELD)
    infer_time = 0
    with torch.no_grad():
        for it, data in enumerate(pbar):
            # Pack data
            if not args.no_cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            inputs = dict()
            inputs['left_rgb'] = data['left_rgb']
            inputs['right_rgb'] = data['right_rgb']
            if cfg.to_disparity:
                inputs['left_sd'] = data['left_sdisp']
                inputs['right_sd'] = data['right_sdisp']
            else:
                inputs['left_sd'] = data['left_sd']
                inputs['right_sd'] = data['right_sd']
            if args.dataset == 'kitti2017':
                target_d = data['left_d']
            target_disp = data['left_disp']
            img_w = data['width'].item()

            # Inference
            end = time.time()
            pred = model(inputs)
            if cfg.to_disparity:
                pred_d = utils.disp2depth(pred, img_w)
                pred_disp = pred
            else:
                raise NotImplementedError
            infer_time += (time.time() - end)

            # Measure performance
            if cfg.to_disparity:
                # disparity
                pred_disp_np = pred_disp.data.cpu().numpy()
                target_disp_np = target_disp.data.cpu().numpy()
                disp_results = disp_meters.compute(pred_disp_np, target_disp_np)
                disp_avg_meters.update(disp_results)
                if args.dataset == 'kitti2017':
                    # depth
                    pred_d_np = pred_d.data.cpu().numpy()
                    target_d_np = target_d.data.cpu().numpy()
                    depth_results = depth_meters.compute(pred_d_np, target_d_np)
                    depth_avg_meters.update(depth_results)
            else:
                raise NotImplementedError
    infer_time /= len(loader)

    if cfg.to_disparity:
        disp_avg_results = disp_avg_meters.compute()
        print('Disparity metric:')
        for key, val in disp_avg_results.items():
            print('- {}: {}'.format(key, val))
    if args.dataset == 'kitti2017':
        depth_avg_results = depth_avg_meters.compute()
        print('Depth metric:')
        for key, val in depth_avg_results.items():
             print('- {}: {}'.format(key, val))
    print('Average infer time: {}'.format(infer_time))


if __name__ == '__main__':
    main()
