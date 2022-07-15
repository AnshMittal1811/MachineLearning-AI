# Prepare MatterportLayout dataset

References:
- [3D Manhattan Room Layout Reconstruction from a Single 360 Image](https://arxiv.org/abs/1910.04099)
- [PanoAnnotator](https://github.com/SunDaDenny/PanoAnnotator)
- [LayoutMP3D: Layout Annotation of Matterport3D](https://arxiv.org/abs/2003.13516)
- [Matterport3DLayoutAnnotation github](https://github.com/ericsujw/Matterport3DLayoutAnnotation) (we use the annotation provided by LayoutNetv2)

## Dataset preparation
### Step 1: download source
Please refer to [Matterport3DLayoutAnnotation](https://github.com/ericsujw/Matterport3DLayoutAnnotation) to download the source datas.
- Put all the rgb under `{ROOT}/image_up/`.
- Download the annotation to `{ROOT}/label_data/` (originally json format).
- Download the data split into `{ROOT}/mp3d_[train|val|test].txt`.

### Step 2: convert json annotation to corners in txt format
Use below code to convert original ground-truth json into txt. **(Remember to update the uppercase variables)**
```python
import os
import glob
import json
import numpy as np

IN_GLOB = 'label_data/*json'
OUT_DIR = 'label_cor'
os.makedirs(OUT_DIR, exist_ok=True)

for p in glob.glob(IN_GLOB):
    gt = json.load(open(p))
    assert gt['cameraHeight'] == 1.6
    us = np.array([pts['coords'][0] for pts in gt['layoutPoints']['points']])
    us = us * 1024
    cs = np.array([pts['xyz'] for pts in gt['layoutPoints']['points']])
    cs = np.sqrt((cs**2)[:, [0, 2]].sum(1))

    vf = np.arctan2(-1.6, cs)
    vc = np.arctan2(-1.6 + gt['layoutHeight'], cs)
    vf = (-vf / np.pi + 0.5) * 512
    vc = (-vc / np.pi + 0.5) * 512

    cor_x = np.repeat(us, 2)
    cor_y = np.stack([vc, vf], -1).reshape(-1)
    cor_xy = np.stack([cor_x, cor_y], -1)

    out_path = os.path.join(OUT_DIR, os.path.split(p)[-1][:-4] + 'txt')
    with open(out_path, 'w') as f:
        for x, y in cor_xy:
            f.write('%.2f %.2f\n' % (x, y))
```

### Step 3: data split
Use below code to organize the data split for training and evaluation. **(Remember to update the uppercase variables)**
```python
import os
from shutil import copy2

IMG_ROOT = 'image_up'
TXT_ROOT = 'label_cor'
OUT_ROOT = 'mp3d_layout'
TRAIN_TXT = 'mp3d_train.txt'
VALID_TXT = 'mp3d_val.txt'
TEST_TXT = 'mp3d_test.txt'

def go(txt, split):
    out_img_root = os.path.join(OUT_ROOT, split, 'img')
    out_txt_root = os.path.join(OUT_ROOT, split, 'label_cor')
    os.makedirs(out_img_root, exist_ok=True)
    os.makedirs(out_txt_root, exist_ok=True)

    with open(txt) as f:
        ks = ['_'.join(l.strip().split()) for l in f]

    for k in ks:
        copy2(os.path.join(IMG_ROOT, k + '.png'), out_img_root)
        copy2(os.path.join(TXT_ROOT, k + '_label.txt'), out_txt_root)
        os.rename(os.path.join(out_txt_root, k + '_label.txt'), os.path.join(out_txt_root, k + '.txt'))


go(TRAIN_TXT, 'train')
go(VALID_TXT, 'valid')
go(TEST_TXT, 'test')
```

### Step 4: clamp occlusion
We assume only visible corners in txt annotation (which is the same as [Holistic 3D Vision Challenge, ECCV2020](https://competitions.codalab.org/competitions/24183#learn_the_details-evaluation)'s format).
For MatterportLayout dataset, please copy&paste below script to `clamp_occ_corners.py` and run:
- `python clamp_occ_corners.py --ori_glob "data/mp3d_layout/train/label_cor/*txt" --output_dir data/mp3d_layout/train_no_occ/label_cor/*txt`
- `python clamp_occ_corners.py --ori_glob "data/mp3d_layout/valid/label_cor/*txt" --output_dir data/mp3d_layout/valid_no_occ/label_cor/*txt`
- `python clamp_occ_corners.py --ori_glob "data/mp3d_layout/test/label_cor/*txt" --output_dir data/mp3d_layout/test_no_occ/label_cor/*txt`
```python
import os
import json
import glob
import numpy as np
from shapely.geometry import LineString

from misc import panostretch

def cor_2_1d(cor, H=512, W=1024):
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
    #bon = ((bon + 0.5) / H - 0.5) * np.pi
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



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ori_glob', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths = glob.glob(args.ori_glob)
    for path in paths:
        if path.endswith('json'):
            with open(path) as f:
                dt = json.load(f)
            cor = np.array(dt['uv'], np.float32)
            cor[:, 0] *= 1024
            cor[:, 1] *= 512
        else:
            with open(path) as f:
                cor = np.array([l.strip().split() for l in f]).astype(np.float32)
        cor = cor.reshape(-1, 4)
        duplicated = [False] * len(cor)
        for i in range(len(duplicated)):
            for j in range(i+1, len(duplicated)):
                if (cor[j] ==  cor[i]).sum() == 4:
                    duplicated[j] = True
        cor = cor[~np.array(duplicated)].reshape(-1, 2)
        cor[:, 0] = cor[:, 0] % 1024
        cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)
        occlusion = find_occlusion(cor[::2].copy()).repeat(2)

        bon = cor_2_1d(cor)

        cor_v1 = []
        for i in range(0, len(cor), 2):
            if occlusion[i] & ~occlusion[(i+2) % len(cor)]:
                cur_x = cor[i, 0]
                next_x = cor[(i+2) % len(cor), 0]
                prev_x, j = None, i-2
                while prev_x is None:
                    if j < 0:
                        j += len(cor)
                    if ~occlusion[j]:
                        prev_x = cor[j, 0]
                        break
                    j -= 2
                dist2next = min(abs(next_x-cur_x), abs(next_x+1024-cur_x), abs(next_x-1024-cur_x))
                dist2prev = min(abs(prev_x-cur_x), abs(prev_x+1024-cur_x), abs(prev_x-1024-cur_x))
                # print(cor[i], prev_x, next_x, dist2next, dist2prev)
                if dist2prev < dist2next:
                    cor_v1.append([prev_x, bon[0, (int(prev_x)+1) % 1024]])
                    cor_v1.append([prev_x, bon[1, (int(prev_x)+1) % 1024]])
                else:
                    cor_v1.append([next_x, bon[0, (int(next_x)-1) % 1024]])
                    cor_v1.append([next_x, bon[1, (int(next_x)-1) % 1024]])
            elif ~occlusion[i]:
                cor_v1.extend(cor[i:i+2])

        cor_v1 = np.stack(cor_v1, 0)
        for _ in range(len(cor_v1)):
            if np.alltrue(cor_v1[::2, 0][1:] - cor_v1[::2, 0][:-1] >= 0):
                break
            cor_v1 = np.roll(cor_v1, 2, axis=0)
        if not np.alltrue(cor_v1[::2, 0][1:] - cor_v1[::2, 0][:-1] >= 0):
            cor_v1[2::2] = np.flip(cor_v1[2::2], 0)
            cor_v1[3::2] = np.flip(cor_v1[3::2], 0)
        for _ in range(len(cor_v1)):
            if np.alltrue(cor_v1[::2, 0][1:] - cor_v1[::2, 0][:-1] >= 0):
                break
            cor_v1 = np.roll(cor_v1, 2, axis=0)
        with open(os.path.join(args.output_dir, f'{os.path.split(path)[1].replace("json", "txt")}'), 'w') as f:
            for u, v in cor_v1:
                f.write(f'{u:.0f} {v:.0f}\n')
```



### Final file structure
So now, you should have a `mp3d_layout` directory with below structure for HoHoNet to train.

	data
    └── mp3d_layout
        ├── train
        │   ├── img/*png
        │   └── label_cor/*txt
        ├── train_no_occ
        │   ├── img/*png
        │   └── label_cor/*txt
        ├── valid
        │   ├── img/*png
        │   └── label_cor/*txt
        ├── valid_no_occ
        │   ├── img/*png
        │   └── label_cor/*txt
        ├── test
        │   ├── img/*png
        │   └── label_cor/*txt
        └── test_no_occ
            ├── img/*png
            └── label_cor/*txt
