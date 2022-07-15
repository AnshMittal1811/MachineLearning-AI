# Prepare Stanford2d3d dataset

## Dataset preparation
### Step 1: download source
Please refer to [2D-3D-Semantics](https://github.com/alexsax/2D-3D-Semantics) to download the source datas.
Make sure `"$S2D3D_ROOT"/area_[1|2|3|4|5a|5b|6]/pano/[depth|rgb|semantic]` existed.


### Step 2: resize and copy into `data/stanford2D3D/` for depth modality
The source data are in high resolution (`2048x4096`).
To reduce data loading time during training, we resize them to `512x1024` and copy into HoHoNet's `data/`.
Copy below code and paste into `prepare_S2D3D_d.py`.
Run `python prepare_S2D3D_d.py --ori_root "$S2D3D_ROOT" --new_root "$HOHO_ROOT/data/stanford2D3D/"`.
```python
import os
import glob
import argparse
from tqdm import tqdm

import numpy as np
from imageio import imread, imwrite
from skimage.transform import rescale

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ori_root', required=True)
parser.add_argument('--new_root', required=True)
args = parser.parse_args()

areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6']

for area in areas:
    print('Processing:', area)
    os.makedirs(os.path.join(args.new_root, area, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.new_root, area, 'depth'), exist_ok=True)
    for fname in tqdm(os.listdir(os.path.join(args.ori_root, area, 'pano', 'rgb'))):
        if fname[0] == '.' or not fname.endswith('png'):
            continue
        rgb_path = os.path.join(args.ori_root, area, 'pano', 'rgb', fname)
        d_path = os.path.join(args.ori_root, area, 'pano', 'depth', fname[:-7] + 'depth.png')
        assert os.path.isfile(d_path)

        rgb = imread(rgb_path)[..., :3]
        depth = imread(d_path)
        rgb = rescale(rgb, 0.25, order=0, mode='wrap', anti_aliasing=False, preserve_range=True)
        depth = rescale(depth, 0.25, order=0, mode='wrap', anti_aliasing=False, preserve_range=True)

        imwrite(os.path.join(args.new_root, area, 'rgb', fname), rgb.astype(np.uint8))
        imwrite(os.path.join(args.new_root, area, 'depth', fname[:-7] + 'depth.png'), depth.astype(np.uint16))
```

### Step 3: resize and copy into `data/s2d3d_sem` for semantic modality
Please download `semantic_labels.json`, `name2label.json`, and `colors.npy` on [Google drive](https://drive.google.com/drive/folders/1raT3vRXnQXRAQuYq36dE-93xFc_hgkTQ?usp=sharing) or [Dropbox](https://www.dropbox.com/sh/b014nop5jrehpoq/AACWNTMMHEAbaKOO1drqGio4a?dl=0).
Put these files under your `$S2D3D_ROOT/`.
Copy below code and paste into `prepare_S2D3D_sem.py`.
Run `python prepare_S2D3D_sem.py --ori_root "$S2D3D_ROOT" --new_root "$HOHO_ROOT/data/s2d3d_sem/"`.
```python
import os
import json
import glob
from PIL import Image
from tqdm import trange
import numpy as np
from shutil import copyfile

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ori_root', required=True)
parser.add_argument('--new_root', required=True)
args = parser.parse_args()

areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6']

with open(os.path.join(args.ori_root, 'semantic_labels.json')) as f:
    id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']

with open(os.path.join(args.ori_root, 'name2label.json')) as f:
    name2id = json.load(f)

colors = np.load(os.path.join(args.ori_root, 'colors.npy'))

id2label = np.array([name2id[name] for name in id2name], np.uint8)

for area in areas:
    rgb_paths = sorted(glob.glob(os.path.join(args.ori_root, area, 'pano', 'rgb', '*png')))
    sem_paths = sorted(glob.glob(os.path.join(args.ori_root, area, 'pano', 'semantic', '*png')))
    os.makedirs(os.path.join(args.new_root, area, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.new_root, area, 'semantic'), exist_ok=True)
    os.makedirs(os.path.join(args.new_root, area, 'semantic_visualize'), exist_ok=True)
    for i in trange(len(rgb_paths)):
        rgb_k = os.path.split(rgb_paths[i])[-1]
        sem_k = os.path.split(sem_paths[i])[-1]

        # RGB
        rgb = Image.open(rgb_paths[i]).convert('RGB').resize((1024, 512), Image.LANCZOS)
        rgb.save(os.path.join(args.new_root, area, 'rgb', rgb_k))
        vis = np.array(rgb)
        # Semantic
        sem = np.array(Image.open(sem_paths[i]).resize((1024, 512), Image.NEAREST), np.int32)
        unk = (sem[..., 0] != 0)
        sem = id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        Image.fromarray(sem).save(os.path.join(args.new_root, area, 'semantic', rgb_k))
        # Visualization
        vis = vis // 2 + colors[sem] // 2
        Image.fromarray(vis).save(os.path.join(args.new_root, area, 'semantic_visualize', rgb_k))
```

### Step 4: prepare data split
Download data split `fold[1|2|3]_[train|valid].txt` and `small_[train|valid|test].txt` on [Google drive](https://drive.google.com/drive/folders/1raT3vRXnQXRAQuYq36dE-93xFc_hgkTQ?usp=sharing) or [Dropbox](https://www.dropbox.com/sh/b014nop5jrehpoq/AACWNTMMHEAbaKOO1drqGio4a?dl=0).
Put these `txt` files under `data/stanford2D3D`.



### Final file structure
So now, you should have a `stanford2D3D` and `s2d3d_sem` directories with below structure for HoHoNet to train.

    data
    ├── stanford2D3D
    │   ├── area_[1|2|3|4|5a|5b|6]
    │   │   ├── img/*png
    │   │   └── depth/*png
    │   ├── small_[train|valid|test].txt
    │   └── fold[1|2|3]_[train|valid].txt
    │
    └── s2d3d_sem
        └── area_[1|2|3|4|5a|5b|6]
            ├── rgb/*png
            └── semantic/*png
