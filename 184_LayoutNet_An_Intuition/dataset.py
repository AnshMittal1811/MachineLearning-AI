import os
import numpy as np
from PIL import Image
from PIL import ImageEnhance

import torch
import torch.utils.data as data


class PanoDataset(data.Dataset):
    '''
    @root_dir (str)
        path to root directory where all data and
        ground truth located at
    @cat_list (list of str)
        list of sub-directories under root_dir to find .png
        e.g. ['img', 'line']
        filenames list of all sub-directories should the same
        i.e.
            if there is a 'room.png' in '{root_dir}/img/',
            '{root_dir}/line/room.png' have to exist
    @flip (bool)
        whether to performe random left-right flip
    @rotate (bool)
        whether to performe random horizontal angle rotate
    @gamma (bool)
        whether to performe random gamma augmentation
        Note that it only perfome on first in cat_list
    '''
    def __init__(self, root_dir, cat_list,
                 flip=False, rotate=False, gamma=False, noise=False, contrast=False,
                 return_filenames=False):
        self.root_dir = root_dir
        self.cat_list = cat_list
        self.fnames = [
            fname for fname in os.listdir(os.path.join(root_dir, cat_list[0]))]
        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.noise = noise
        self.contrast = contrast
        self.return_filenames = return_filenames

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.fnames:
            for cat in self.cat_list:
                cat_path = os.path.join(self.root_dir, cat, fname)
                assert os.path.isfile(cat_path), '%s not found !!!' % cat_path

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path_list = [
            os.path.join(self.root_dir, cat, self.fnames[idx])
            for cat in self.cat_list]
        pilimg_list = [Image.open(path) for path in path_list]
        if self.contrast:
            p = np.random.uniform(0.5, 2)
            pilimg_list = [ImageEnhance.Contrast(pil_img).enhance(p)
                           for pil_img in pilimg_list]
        npimg_list = [np.array(pil_img, np.float32) / 255 for pil_img in pilimg_list]

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            npimg_list = [np.flip(npimg, axis=1) for npimg in npimg_list]

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(npimg_list[0].shape[1])
            npimg_list = [np.roll(npimg, dx, axis=1) for npimg in npimg_list]

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(0.5, 2)
            npimg_list[0] = npimg_list[0] ** p

        # Random noise augmentation
        if self.noise:
            noise = np.random.randn(*npimg_list[0].shape) * 0.05
            npimg_list[0] = npimg_list[0] + noise

        # Transpose to C x H x W
        npimg_list = [
            np.expand_dims(npimg, axis=0) if npimg.ndim == 2 else npimg.transpose([2, 0, 1])
            for npimg in npimg_list]

        if self.return_filenames:
            return tuple(torch.FloatTensor(npimg) for npimg in npimg_list) + \
                (self.fnames[idx], )
        return tuple(torch.FloatTensor(npimg) for npimg in npimg_list)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='data/train')
    parser.add_argument('--cat_list', default=['img', 'line', 'edge', 'cor'], nargs='+')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--rotate', action='store_true')
    args = parser.parse_args()

    print('args:')
    for key, val in vars(args).items():
        print('    {:16} {}'.format(key, val))

    dataset = PanoDataset(
        root_dir=args.root_dir, cat_list=args.cat_list,
        flip=args.flip, rotate=args.rotate)
    print('len(dataset): {}'.format(len(dataset)))

    for ith, x in enumerate(dataset[0]):
        print(
            'size', x.size(),
            '| dtype', x.dtype,
            '| mean', x.mean().item(),
            '| std', x.std().item(),
            '| min', x.min().item(),
            '| max', x.max().item())
