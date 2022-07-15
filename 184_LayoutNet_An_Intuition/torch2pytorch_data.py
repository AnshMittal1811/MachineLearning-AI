import os
import glob
from torch.utils.serialization import load_lua
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve
from pano import find_N_peaks

from PIL import Image

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ORGIN_DATA_DIR = os.path.join(DATA_DIR, 'origin', 'data')
ORGIN_GT_DIR = os.path.join(DATA_DIR, 'origin', 'gt')


# Variables for train/val/test split
cat_list = ['img', 'line', 'edge', 'cor']
train_pats = [
    'panoContext_%s_train.t7',
    'stanford2d-3d_%s_area_1.t7', 'stanford2d-3d_%s_area_2.t7',
    'stanford2d-3d_%s_area_4.t7', 'stanford2d-3d_%s_area_6.t7']
valid_pats = ['panoContext_%s_val.t7', 'stanford2d-3d_%s_area_3.t7']
test_pats = ['panoContext_%s_test.t7', 'stanford2d-3d_%s_area_5.t7']

train_pano_map = os.path.join(DATA_DIR, 'panoContext_trainmap.txt')
valid_pano_map = os.path.join(DATA_DIR, 'panoContext_valmap.txt')
test_pano_map = os.path.join(DATA_DIR, 'panoContext_testmap.txt')


def cvt2png(target_dir, patterns, pano_map_path):
    os.makedirs(target_dir, exist_ok=True)
    for cat in cat_list:
        for pat in patterns:
            # Define source file paths
            th_path = os.path.join(ORGIN_DATA_DIR, pat % cat)
            assert os.path.isfile(th_path), '%s not found !!!' % th_path

            if pat.startswith('stanford'):
                gt_path = os.path.join(
                    ORGIN_GT_DIR, 'pano_id_%s.txt' % pat[-9:-3])
            else:
                gt_path = os.path.join(
                    ORGIN_GT_DIR, 'panoContext_%s.txt' % pat.split('_')[-1].split('.')[0])
            assert os.path.isfile(gt_path), '%s not found !!!' % gt_path

            # Parse file names from gt list
            with open(gt_path) as f:
                fnames = [line.strip() for line in f]
            print('%-30s: %3d examples' % (pat % cat, len(fnames)))

            # Remapping panoContext filenames
            if pat.startswith('pano'):
                fnames_cnt = dict([(v, 0) for v in fnames])
                with open(pano_map_path) as f:
                    for line in f:
                        v, k, _ = line.split()
                        k = int(k)
                        fnames[k] = v
                        fnames_cnt[v] += 1
                for v in fnames_cnt.values():
                    assert v == 1

            # Parse th file
            imgs = load_lua(th_path).numpy()
            assert imgs.shape[0] == len(fnames), 'number of data and gt mismatched !!!'

            # Dump each images to target direcotry
            target_cat_dir = os.path.join(target_dir, cat)
            os.makedirs(target_cat_dir, exist_ok=True)
            for img, fname in zip(imgs, fnames):
                target_path = os.path.join(target_cat_dir, fname)
                if img.shape[0] == 3:
                    # RGB
                    Image.fromarray(
                        (img.transpose([1, 2, 0]) * 255).astype(np.uint8)).save(target_path)
                else:
                    # Gray
                    Image.fromarray(
                        (img[0] * 255).astype(np.uint8)).save(target_path)


cvt2png(os.path.join(DATA_DIR, 'train'), train_pats, train_pano_map)
cvt2png(os.path.join(DATA_DIR, 'valid'), valid_pats, valid_pano_map)
cvt2png(os.path.join(DATA_DIR, 'test'), test_pats, test_pano_map)

# Convert ground truth corner map to corner index
for split in ['test', 'valid', 'train']:
    target_dir = os.path.join(DATA_DIR, split, 'label_cor')
    os.makedirs(target_dir, exist_ok=True)

    # Kernel used to extract corner position from gt corner map
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])

    for path in glob.glob('data/%s/cor/*' % split):
        img_cor = np.array(Image.open(path))
        signal = (img_cor == 255).astype(np.int)
        signal = convolve(signal, kernel, mode='constant', cval=0.0)
        if (signal == 5).sum() != 8:
            cor_id = []
            X_loc = find_N_peaks(signal.sum(0), prominence=None,
                                 distance=20, N=4)[0]
            for x in X_loc:
                x = int(np.round(x))
                V_signal = signal[:, max(0, x-15):x+15].sum(1)
                y1, y2 = find_N_peaks(V_signal, prominence=None,
                                      distance=20, N=2)[0]
                cor_id.append((x, y1))
                cor_id.append((x, y2))
            cor_id = np.array(cor_id)
        else:
            cor_id = np.array([(x, y) for x, y in zip(*np.where((signal.T == 5)))])

        # Arange corner in order if needed
        for i in range(1, len(cor_id), 2):
            if cor_id[i, 1] < cor_id[i-1, 1]:
                cor_id[[i-1, i]] = cor_id[[i, i-1]]
            cor_id[[i-1, i], 0] = cor_id[[i-1, i], 0].mean()

        basename = path.split('/')[-1].split('.')[0]
        with open(os.path.join(target_dir, '%s.txt' % basename), 'w') as f:
            for x, y in cor_id:
                f.write('%d %d\n' % (x, y))
