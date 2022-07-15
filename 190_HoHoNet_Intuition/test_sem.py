import os
import argparse
import importlib
from natsort import natsorted
from tqdm import tqdm, trange
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.config import config, update_config, infer_exp_id
from lib import dataset


if __name__ == '__main__':

    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--pth')
    parser.add_argument('--out')
    parser.add_argument('--vis_dir')
    parser.add_argument('--y', action='store_true')
    parser.add_argument('--test_hw', type=int, nargs='*')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    device = 'cuda' if config.cuda else 'cpu'

    if config.cuda and config.cuda_benchmark:
        torch.backends.cudnn.benchmark = False

    # Init global variable
    if not args.pth:
        from glob import glob
        exp_id = infer_exp_id(args.cfg)
        exp_ckpt_root = os.path.join(config.ckpt_root, exp_id)
        args.pth = natsorted(glob(os.path.join(exp_ckpt_root, 'ep*pth')))[-1]
        print(f'No pth given,  inferring the trained pth: {args.pth}')

    if not args.out:
        args.out = os.path.splitext(args.pth)[0]
        print(f'No out given,  inferring the output dir: {args.out}')
        os.makedirs(args.out, exist_ok=True)
    if os.path.isfile(os.path.join(args.out, 'cm.npz')) and not args.y:
        print(f'{os.path.join(args.out, "cm.npz")} is existed:')
        cm = np.load(os.path.join(args.out, 'cm.npz'))['cm']
        inter = np.diag(cm)
        union = cm.sum(0) + cm.sum(1) - inter
        ious = inter / union
        accs = inter / cm.sum(1)
        DatasetClass = getattr(dataset, config.dataset.name)
        config.dataset.valid_kwargs.update(config.dataset.common_kwargs)
        valid_dataset = DatasetClass(**config.dataset.valid_kwargs)
        id2class = np.array(valid_dataset.ID2CLASS)
        for name, iou, acc in zip(id2class, ious, accs):
            print(f'{name:20s}:    iou {iou*100:5.2f}    /    acc {acc*100:5.2f}')
        print(f'{"Overall":20s}:    iou {ious.mean()*100:5.2f}    /    acc {accs.mean()*100:5.2f}')
        print('Re-write this results ?', end=' ')
        input()

    # Init dataset
    DatasetClass = getattr(dataset, config.dataset.name)
    config.dataset.valid_kwargs.update(config.dataset.common_kwargs)
    if args.test_hw:
        input_hw = config.dataset.common_kwargs['hw']
        config.dataset.valid_kwargs['hw'] = args.test_hw
    else:
        input_hw = None
    valid_dataset = DatasetClass(**config.dataset.valid_kwargs)
    valid_loader = DataLoader(valid_dataset, 1,
                              num_workers=config.num_workers,
                              pin_memory=config.cuda)

    # Init network
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs).to(device)
    net.load_state_dict(torch.load(args.pth))
    net = net.to(device).eval()

    # Start eval
    cm = 0
    num_classes = config.model.kwargs.modalities_config.SemanticSegmenter.num_classes
    with torch.no_grad():
        for batch in tqdm(valid_loader, position=1, total=len(valid_loader)):
            color = batch['x'].to(device)
            sem = batch['sem'].to(device)
            mask = (sem >= 0)
            if mask.sum() == 0:
                continue

            # feed forward & compute losses
            if input_hw is not None:
                color = F.interpolate(color, size=input_hw, mode='bilinear', align_corners=False)
            pred_sem = net.infer(color)['sem']
            if input_hw is not None:
                pred_sem = F.interpolate(pred_sem, size=args.test_hw, mode='bilinear', align_corners=False)

            # Visualization
            if args.vis_dir:
                import matplotlib.pyplot as plt
                from imageio import imwrite
                cmap = (plt.get_cmap('gist_rainbow')(np.arange(num_classes) / num_classes)[...,:3] * 255).astype(np.uint8)
                rgb = (batch['x'][0, :3].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8)
                vis_sem = cmap[pred_sem[0].argmax(0).cpu().numpy()]
                vis_sem = (rgb * 0.2 + vis_sem * 0.8).astype(np.uint8)
                imwrite(os.path.join(args.vis_dir, batch['fname'][0].strip()), vis_sem)
                vis_sem = cmap[sem[0].cpu().numpy()]
                vis_sem = (rgb * 0.2 + vis_sem * 0.8).astype(np.uint8)
                imwrite(os.path.join(args.vis_dir, batch['fname'][0].strip() + '.gt.png'), vis_sem)

            # Log
            gt = sem[mask]
            pred = pred_sem.argmax(1)[mask]
            assert gt.min() >= 0 and gt.max() < num_classes and pred_sem.shape[1] == num_classes
            cm += np.bincount((gt * num_classes + pred).cpu().numpy(), minlength=num_classes**2)

    # Summarize
    print('  Summarize  '.center(50, '='))
    cm = cm.reshape(num_classes, num_classes)
    id2class = np.array(valid_dataset.ID2CLASS)
    valid_mask = (cm.sum(1) != 0)
    cm = cm[valid_mask][:, valid_mask]
    id2class = id2class[valid_mask]
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    ious = inter / union
    accs = inter / cm.sum(1)
    for name, iou, acc in zip(id2class, ious, accs):
        print(f'{name:20s}:    iou {iou*100:5.2f}    /    acc {acc*100:5.2f}')
    print(f'{"Overall":20s}:    iou {ious.mean()*100:5.2f}    /    acc {accs.mean()*100:5.2f}')
    np.savez(os.path.join(args.out, 'cm.npz'), cm=cm)

