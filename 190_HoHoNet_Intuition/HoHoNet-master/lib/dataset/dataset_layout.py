import os
import numpy as np
from PIL import Image
from shapely.geometry import LineString

import torch
import torch.utils.data as data

from lib.misc import panostretch


class PanoCorBonDataset(data.Dataset):

    def __init__(self, root_dir,
                 flip=False, rotate=False, gamma=False, stretch=False,
                 max_stretch=1.5):
        self.img_dir = os.path.join(root_dir, 'img')
        self.cor_dir = os.path.join(root_dir, 'label_cor')
        self.img_fnames = sorted([
            fname for fname in os.listdir(self.img_dir)
            if fname.endswith('.jpg') or fname.endswith('.png')
        ])
        self.txt_fnames = ['%s.txt' % fname[:-4] for fname in self.img_fnames]
        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.stretch = stretch
        self.max_stretch = max_stretch

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.txt_fnames:
            gt_path = os.path.join(self.cor_dir, fname)
            assert os.path.isfile(gt_path),\
                '%s not found' % gt_path
            cor = np.loadtxt(gt_path)
            assert ((cor[:,0] < 0) | (cor[:,0] >= 1024)).sum() == 0, f'coor_x out of range {fname}'
            assert ((cor[:,1] < 0) | (cor[:,1] >= 512)).sum() == 0, 'coor_y out of range'
            assert (cor[0::2,0] != cor[1::2,0]).sum() == 0, 'ceiling-floor x inconsist'
            assert (cor[::2,0][1:] - cor[::2,0][:-1] < 0).sum() == 0, 'format incorrect'

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])
        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
        H, W = img.shape[:2]

        # Read ground truth corners
        with open(os.path.join(self.cor_dir,
                               self.txt_fnames[idx])) as f:
            cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

        # Stretch augmentation
        if self.stretch:
            xmin, ymin, xmax, ymax = cor2xybound(cor)
            kx = np.random.uniform(1.0, self.max_stretch)
            ky = np.random.uniform(1.0, self.max_stretch)
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))
            img, cor = panostretch.pano_stretch(img, cor, kx, ky)

        # Prepare 1d ceiling-wall/floor-wall boundary
        bon = cor_2_1d(cor, H, W)

        # Prepare lrub and occ
        corx = np.round(cor[::2,0])
        xs = []
        lrub = []
        occ = []
        for i in range(len(corx)):
            u, b = cor[[i*2,i*2+1], 1]
            if corx[i] == corx[i-1]:
                lrub[-1][2] = u
                lrub[-1][3] = b
                occ[-1] = 1
            else:
                xs.append(cor[i*2, 0])
                lrub.append([u, b, u, b])
                occ.append(0)
        xs = np.array(xs)
        lrub = np.array(lrub)
        occ = np.array(occ)
        uidx = np.arange(W)
        cdist = np.abs(uidx[:,None] - xs[None,:])
        cdist[cdist > W/2] = W - cdist[cdist > W/2]
        nnidx = cdist.argmin(1)
        lrub = lrub[nnidx].T
        lrub = ((lrub + 0.5) / H - 0.5) * np.pi
        occ = occ[nnidx][None]

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            bon = np.flip(bon, axis=1)
            lrub = np.flip(lrub, axis=1)
            occ = np.flip(occ, axis=1)
            lrub = np.concatenate([lrub[2:], lrub[:2]], 0)
            cor[:, 0] = img.shape[1] - 1 - cor[:, 0]

        # Random horizontal rotate
        if self.rotate:
            dx_p = np.random.choice([0, 0.25, 0.5, 0.75])
            dx = int(img.shape[1] * dx_p)
            img = np.roll(img, dx, axis=1)
            bon = np.roll(bon, dx, axis=1)
            lrub = np.roll(lrub, dx, axis=1)
            occ = np.roll(occ, dx, axis=1)
            cor[:, 0] = (cor[:, 0] + dx) % img.shape[1]

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1., 2.)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img ** p

        # Prepare 1d wall-wall probability
        uidx = np.arange(W)
        corx = np.round(np.unique(cor[:, 0])).astype(np.int32)
        corx = np.concatenate([corx, corx+W, corx-W])
        dist = corx[:,None] - uidx[None,:]
        vot = dist[None, np.abs(dist).argmin(0), uidx]

        # Convert all data to tensor
        out_dict = {
            'x': torch.FloatTensor(img.transpose([2, 0, 1]).copy()),
            'bon': torch.FloatTensor(bon.copy()),
            'vot': torch.FloatTensor(vot.copy()),
            'lrub': torch.FloatTensor(lrub.copy()),
            'occ': torch.FloatTensor(occ.copy()),
            'img_path': img_path,
        }

        return out_dict


def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def find_occlusion(coor):
    u = panostretch.coorx2u(coor[:, 0])
    v = panostretch.coory2v(coor[:, 1])
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i+1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)


def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostretch.coorx2u(corU[:, 0])
    vU = panostretch.coory2v(corU[:, 1])
    vB = panostretch.coory2v(corB[:, 1])

    x, y = panostretch.uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)


def visualize_a_data(x, y_bon, y_cor):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    y_bon = y_bon.numpy()
    y_bon = ((y_bon / np.pi + 0.5) * x.shape[0]).round().astype(int)
    y_cor = y_cor.numpy()

    gt_cor = np.zeros((30, 1024, 3), np.uint8)
    gt_cor[:] = y_cor[0][None, :, None] * 255
    img_pad = np.zeros((3, 1024, 3), np.uint8) + 255

    img_bon = (x.copy() * 0.5).astype(np.uint8)
    y1 = np.round(y_bon[0]).astype(int)
    y2 = np.round(y_bon[1]).astype(int)
    y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    img_bon[y_bon[0], np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1], np.arange(len(y_bon[1])), 1] = 255

    return np.concatenate([gt_cor, img_pad, img_bon], 0)


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

