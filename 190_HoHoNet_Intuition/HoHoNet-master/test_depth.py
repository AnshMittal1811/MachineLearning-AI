import os
import argparse
import importlib
from natsort import natsorted
from tqdm import tqdm, trange
from collections import Counter

import numpy as np
from imageio import imwrite
from scipy.spatial.transform import Rotation
from lib.misc.pano_lsd_align import rotatePanorama, panoEdgeDetection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.config import config, update_config, infer_exp_id
from lib import dataset


def eval_metric(pred, gt, dmax):
    gt = gt.clamp(0.01, dmax)
    pred = pred.clamp(0.01, dmax)
    mre = ((gt - pred).abs() / gt).mean().item()
    mae = (gt - pred).abs().mean().item()
    rmse = ((gt - pred)**2).mean().sqrt().item()
    rmse_log = ((gt.log10() - pred.log10())**2).mean().sqrt().item()
    log10 = (gt.log10() - pred.log10()).abs().mean().item()

    delta = torch.max(pred/gt, gt/pred)
    delta_1 = (delta < 1.25).float().mean().item()
    delta_2 = (delta < 1.25**2).float().mean().item()
    delta_3 = (delta < 1.25**3).float().mean().item()
    return {
        'mre': mre, 'mae': mae, 'rmse': rmse, 'rmse_log': rmse_log, 'log10': log10,
        'delta_1': delta_1, 'delta_2': delta_2, 'delta_3': delta_3,
    }


if __name__ == '__main__':

    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--pth')
    parser.add_argument('--out')
    parser.add_argument('--vis_dir')
    parser.add_argument('--clip', default=10, type=float)
    parser.add_argument('--y', action='store_true')
    parser.add_argument('--pitch', default=0, type=float)
    parser.add_argument('--roll', default=0, type=float)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    device = 'cuda' if config.cuda else 'cpu'

    if not args.pth:
        from glob import glob
        exp_id = infer_exp_id(args.cfg)
        exp_ckpt_root = os.path.join(config.ckpt_root, exp_id)
        args.pth = natsorted(glob(os.path.join(exp_ckpt_root, 'ep*pth')))[-1]
        print(f'No pth given,  inferring the trained pth: {args.pth}')

    if not args.out:
        out = [os.path.splitext(args.pth)[0]]
        if args.pitch > 0:
            out.append(f'.pitch{args.pitch:.0f}')
        if args.roll > 0:
            out.append(f'.roll{args.roll:.0f}')
        args.out = ''.join(out + ['.npz'])
        print(f'No out given,  inferring the output path: {args.out}')
    if os.path.isfile(args.out) and not args.y:
        print(f'{args.out} is existed:')
        print(dict(np.load(args.out)))
        print('Re-write this results ?', end=' ')
        input()

    # Init dataset
    DatasetClass = getattr(dataset, config.dataset.name)
    config.dataset.valid_kwargs.update(config.dataset.common_kwargs)
    config.dataset.valid_kwargs['fix_pitch'] = args.pitch
    config.dataset.valid_kwargs['fix_roll'] = args.roll
    valid_dataset = DatasetClass(**config.dataset.valid_kwargs)

    # Init network
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs).to(device)
    net.load_state_dict(torch.load(args.pth))
    net.eval()

    # Run evaluation
    evaluation_metric = Counter()
    for batch in tqdm(valid_dataset):
        # Add batch dim and move to gpu
        color = batch['x'][None].to(device)
        depth = batch['depth'][None].to(device)
        mask = (depth > 0)

        # feed forward
        with torch.no_grad():
            pred_depth = net.infer(color)
        if not torch.is_tensor(pred_depth):
            viz_dict = pred_depth
            pred_depth = viz_dict.pop('depth')
        pred_depth = pred_depth.clamp(0.01)

        if args.pitch:
            vp = Rotation.from_rotvec([-args.pitch * np.pi / 180, 0, 0]).as_matrix()
            pred_depth = pred_depth.squeeze()[...,None].cpu().numpy()
            pred_depth = rotatePanorama(pred_depth, vp, order=0)[...,0]
            pred_depth = torch.from_numpy(pred_depth[None,None]).to(depth.device)
        if args.roll:
            vp = Rotation.from_rotvec([0, -args.roll * np.pi / 180, 0]).as_matrix()
            pred_depth = pred_depth.squeeze()[...,None].cpu().numpy()
            pred_depth = rotatePanorama(pred_depth, vp, order=0)[...,0]
            pred_depth = torch.from_numpy(pred_depth[None,None]).to(depth.device)

        if args.vis_dir:
            fname = batch['fname'].strip()
            os.makedirs(args.vis_dir, exist_ok=True)
            rgb = (batch['x'].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8)
            dep = pred_depth.squeeze().mul(512).cpu().numpy().astype(np.uint16)
            dep[~mask.squeeze().cpu().numpy()] = 0
            gtdep = depth.squeeze().mul(512).cpu().numpy().astype(np.uint16)
            imwrite(os.path.join(args.vis_dir, fname + '.rgb' + '.jpg'), rgb)
            imwrite(os.path.join(args.vis_dir, fname + '.rgb' + '.png'), gtdep)
            imwrite(os.path.join(args.vis_dir, fname + '.depth' + '.png'), dep)
            for k, v in viz_dict.items():
                if v.dtype == np.uint8 or v.dtype == np.uint16:
                    imwrite(os.path.join(args.vis_dir, fname + '.' + k + '.png'), v)
                else:
                    raise NotImplementedError

        evaluation_metric['N'] += 1
        for metric, v in eval_metric(pred_depth[mask], depth[mask], args.clip).items():
            evaluation_metric[metric] += v

    N = evaluation_metric.pop('N')
    for metric, v in evaluation_metric.items():
        evaluation_metric[metric] = v / N
    for metric, v in evaluation_metric.items():
        print(f'{metric:20s} {v:.4f}')

    np.savez(args.out, **evaluation_metric)

