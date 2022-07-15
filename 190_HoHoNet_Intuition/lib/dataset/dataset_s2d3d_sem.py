import os
import glob
import numpy as np
from imageio import imread
from shapely.geometry import LineString

import torch
import torch.utils.data as data
import torch.nn.functional as F

from lib.misc import panostretch

__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_valid': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_valid': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_valid': ['area_1', 'area_3', 'area_6'],
}

class S2d3dSemDataset(data.Dataset):
    NUM_CLASSES = 13
    ID2CLASS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
    def __init__(self, root, fold, depth=True, hw=(512, 1024), mask_black=True, flip=False, rotate=False):
        assert fold in __FOLD__, 'Unknown fold'
        self.depth = depth
        self.hw = hw
        self.mask_black = mask_black
        self.rgb_paths = []
        self.sem_paths = []
        self.dep_paths = []
        for dname in __FOLD__[fold]:
            self.rgb_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))
            self.sem_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'semantic', '*png'))))
            self.dep_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'depth', '*png'))))
        assert len(self.rgb_paths)
        assert len(self.rgb_paths) == len(self.sem_paths)
        assert len(self.rgb_paths) == len(self.dep_paths)
        self.flip = flip
        self.rotate = rotate

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = torch.FloatTensor(imread(self.rgb_paths[idx]) / 255.).permute(2,0,1)
        sem = torch.LongTensor(imread(self.sem_paths[idx])) - 1
        if self.depth:
            dep = imread(self.dep_paths[idx])
            dep = np.where(dep==65535, 0, dep/512)
            dep = np.clip(dep, 0, 4)
            dep = torch.FloatTensor(dep[None])
            rgb = torch.cat([rgb, dep], 0)
        H, W = rgb.shape[1:]
        if (H, W) != self.hw:
            rgb = F.interpolate(rgb[None], size=self.hw, mode='bilinear', align_corners=False)[0]
            sem = F.interpolate(sem[None,None].float(), size=self.hw, mode='nearest')[0,0].long()

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            rgb = torch.flip(rgb, (-1,))
            sem = torch.flip(sem, (-1,))

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(W)
            rgb = torch.roll(rgb, dx, dims=-1)
            sem = torch.roll(sem, dx, dims=-1)

        # Mask out top-down black
        if self.mask_black:
            sem[rgb.sum(0) == 0] = -1

        # Convert all data to tensor
        out_dict = {
            'x': rgb,
            'sem': sem,
            'fname': os.path.split(self.rgb_paths[idx])[1].ljust(200),
        }
        return out_dict


if __name__ == '__main__':

    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='data/valid/')
    parser.add_argument('--ith', default=0, type=int,
                        help='Pick a data id to visualize.'
                             '-1 for visualize all data')
    parser.add_argument('--flip', action='store_true',
                        help='whether to random flip')
    parser.add_argument('--rotate', action='store_true',
                        help='whether to random horizon rotation')
    parser.add_argument('--gamma', action='store_true',
                        help='whether to random luminance change')
    parser.add_argument('--stretch', action='store_true',
                        help='whether to random pano stretch')
    parser.add_argument('--dist_clip', default=20)
    parser.add_argument('--out_dir', default='data/vis_dataset')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('args:')
    for key, val in vars(args).items():
        print('    {:16} {}'.format(key, val))

    dataset = PanoCorBonDataset(
        root_dir=args.root_dir,
        flip=args.flip, rotate=args.rotate, gamma=args.gamma, stretch=args.stretch)

    # Showing some information about dataset
    print('len(dataset): {}'.format(len(dataset)))
    batch = dataset[args.ith]
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(k, v.shape)
        else:
            print(k, v)
    print('=' * 20)

    if args.ith >= 0:
        to_visualize = [dataset[args.ith]]
    else:
        to_visualize = dataset

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('bwr')
    for batch in tqdm(to_visualize):
        fname = os.path.split(batch['img_path'])[-1]
        img = batch['x'].permute(1,2,0).numpy()
        y_bon = batch['bon'].numpy()
        y_bon = ((y_bon / np.pi + 0.5) * img.shape[0]).round().astype(int)
        img[y_bon[0], np.arange(len(y_bon[0])), 1] = 1
        img[y_bon[1], np.arange(len(y_bon[1])), 1] = 1
        img = (img * 255).astype(np.uint8)
        img_pad = np.full((3, 1024, 3), 255, np.uint8)
        img_vot = batch['vot'].repeat(30, 1).numpy()
        img_vot = (img_vot / args.dist_clip + 1) / 2
        vot_mask = (img_vot >= 0) & (img_vot <= 1)
        img_vot = (cmap(img_vot)[...,:3] * 255).astype(np.uint8)
        img_vot[~vot_mask] = 0
        out = np.concatenate([img_vot, img_pad, img], 0)
        Image.fromarray(out).save(os.path.join(args.out_dir, fname))

