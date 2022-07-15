# Reproduction

Below provides:
1. guide to prepare the datasets for each task in our paper
2. reproduce the training and numerical results in our paper

## Dataset
Detail instruction for preparing the datas for each dataset and task:
- `Matterport3d` x `Layout`
    - see [Prepare MatterportLayout dataset](README_prepare_data_mp3d_layout.md)
- `Matterport3d` x `Depth (BiFuse's stitching)`
    - We use the rgb-d stitching provided by [BiFuse](https://github.com/Yeh-yu-hsuan/BiFuse)
    - Put their `mp3d_align/` under `data/`
    - Download data split via [Google drive](https://drive.google.com/drive/folders/1raT3vRXnQXRAQuYq36dE-93xFc_hgkTQ?usp=sharing) or via [Dropbox](https://www.dropbox.com/sh/b014nop5jrehpoq/AACWNTMMHEAbaKOO1drqGio4a?dl=0) and put them under `data/matterport3d/`.
- `Matterport3d` x `Depth (our new stitching)`
    - We remove the depth noise in BiFuse's stitching
    - This is not the version we use in our paper
    - **TODO:** release new stiching code with experiment results on it
- `Stanford2d3d` x `Depth`:
    - see [Prepare Stanford2d3d dataset](README_prepare_data_s2d3d.md)
- `Stanford2d3d` x `Semantic segmentation`:
    - see [Prepare Stanford2d3d dataset](README_prepare_data_s2d3d.md)

The overall file strucure of the datasets is depicted as follow:

    data
    ├── mp3d_align                 # Stitching provided by BiFuse (https://github.com/Yeh-yu-hsuan/BiFuse)
    │   ├── 17DRP5sb8fy
    │   │   ├── 00ebbf3782c64d74aaf7dd39cd561175
    │   │   │   ├── color.jpg
    │   │   │   └── depth.npy
    │   │   └── ...
    │   └── ...
    │
    ├── matterport3d
    │   ├── scenes_abla_train.txt  # 41 house id for ablation training
    │   ├── scenes_abla_valid.txt  # 20 house id for ablation evaluation
    │   ├── scenes_train.txt       # 61 house id for training following BiFuse
    │   ├── mp3d_scenes_test.txt   # 28 house id for testing following BiFuse
    │   └── mp3d_rgbd/             # Our new stitching which fixs the depth noise in BiFuse's version
    │                              # Release new stitching code with new experiments later.
    │
    ├── mp3d_layout                # Please follow README_prepare_data_mp3d_layout.md
    │   ├── train_no_occ
    │   │   ├── img/*png
    │   │   └── label_cor/*txt
    │   ├── valid_no_occ
    │   │   ├── img/*png
    │   │   └── label_cor/*txt
    │   └── test_no_occ
    │       ├── img/*png
    │       └── label_cor/*txt
    │
    ├── stanford2D3D               # Please follow README_prepare_data_s2d3d.md
    │   ├── area_[1|2|3|4|5a|5b|6]
    │   │   ├── img/*png
    │   │   └── depth/*png
    │   ├── small_[train|valid|test].txt
    │   └── fold[1|2|3]_[train|valid].txt
    │
    └── s2d3d_sem                  # Please follow README_prepare_data_s2d3d.md
        └── area_[1|2|3|4|5a|5b|6]
            ├── rgb/*png
            └── semantic/*png


## Reproduction: training
The configs for reproducing the experiments are all in `config/`.

Just run:
```
python train.py --cfg {PATH_TO_CONFIG}
```
to train the same setting as experiments in our paper.
Note that the results with same config but different runs could be different as the random seed is not fixed.

Some examples:
```
python train.py --cfg config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml
python train.py --cfg config/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34.yaml
python train.py --cfg config/s2d3d_depth/HOHO_depth_dct_efficienthc_TransEn1.yaml
python train.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h1024_fold1_resnet101.yaml
```

## Reproduction: measuring FPS
Just run:
```
python count_params_flops.py --cfg {PATH_TO_CONFIG}
```
It measures averaged feed-forward times of the model.
The results reported in our paper are obtained on a GeForce RTX 2080.

## Reproduction: quantitative evaluation
Please make sure the dataset and the trained weights are organized as the instruction above.
If not, the config should be updated accordinly and you should directly assign the path to the trained weight to the testing script via `--pth`.


<br/>

### `Matterport3D` x `depth` (BiFuse's stitching and setting)
Assume pretrained weights located at:
- `ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth`
- `ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1/ep60.pth`

Run:
```
python test_depth.py --cfg config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1.yaml
python test_depth.py --cfg config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml
```

Results:
| Exp | fps | mre | mae | rmse | rmse_log | log10 | delta_1 | delta_3 | delta_3 |
| :-- | :-- | :-- | :-- | :--- | :------- | :---- | :------ | :------ | :------ |
| HOHO_depth_dct_efficienthc_TransEn1 | 52 | 0.1488 | 0.2862 | 0.5138 | 0.0871 | 0.0505 | 0.8786 | 0.9519 | 0.9771 |
| HOHO_depth_dct_efficienthc_TransEn1_hardnet | 67 | 0.1482 | 0.2761 | 0.4968 | 0.0857 | 0.0494 | 0.8830 | 0.9547 | 0.9797 |


<br/>

### `Matterport3D` x `depth` (our new stitching and setting)
**TODO**


<br/>

### `Matterport3D` x `layout` (LayoutNetv2's setting)
Assume pretrained weights located at:
- `ckpt/mp3d_layout_HOHO_layout_aug_efficienthc_Transen1_resnet34/ep300.pth`

Run to predict layout and store the results in txt files:
```
python test_layout.py --cfg config/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34.yaml --img_glob "data/mp3d_layout/test/img/*" --output_dir output/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34/
```

Run to evaluate the prediction:
```
python eval_layout.py --gt_glob "data/mp3d_layout/test/label_cor/*" --dt_glob "output/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34/*"
```

Results:
| Exp | fps | 2DIoU | 3DIoU | RMSE | delta_1 |
| :-- | :-- | :---- | :---- | :--- | :------ |
| HOHO_layout_aug_efficienthc_Transen1_resnet34 | 111 | 82.32 | 79.88 | 0.22 | 0.95 |

**[Note]** our implementation for the depth-based evaluation (i.e., RMSE, delta_1) is very different from LayoutNetv2's so the results from the two repo is not direct comparable.


<br/>

### `Stanford2d3d` x `depth` (BiFuse's setting)
Assume pretrained weights located at:
- `ckpt/s2d3d_depth_HOHO_depth_dct_efficienthc_TransEn1/ep60.pth`

Run:
```
python test_depth.py --cfg config/s2d3d_depth/HOHO_depth_dct_efficienthc_TransEn1.yaml
```

Results:
| Exp | fps | mre | mae | rmse | rmse_log | log10 | delta_1 | delta_3 | delta_3 |
| :-- | :-- | :-- | :-- | :--- | :------- | :---- | :------ | :------ | :------ |
| HOHO_depth_dct_efficienthc_TransEn1 | 52 | 0.1014 | 0.2027 | 0.3834 | 0.0668 | 0.0438 | 0.9054 | 0.9693 | 0.9886 |


<br/>

### `Stanford2d3d` x `depth` (GeoReg360's setting)
Assume pretrained weights located at:
- `ckpt/s2d3d_depth_HOHO_depthS_dct_efficienthc_TransEn1/ep60.pth`
- `ckpt/s2d3d_depth_HOHO_depthS_SGD_dct_efficienthc_TransEn1/ep60.pth`

Run:
```
python test_depth.py --cfg config/s2d3d_depth/HOHO_depthS_SGD_dct_efficienthc_TransEn1.yaml --clip 100
python test_depth.py --cfg config/s2d3d_depth/HOHO_depthS_dct_efficienthc_TransEn1.yaml --clip 100
```

**[Note]** remember to add `--clip 100` to disable depth clip for a fair comparison with GeoReg360's setting.

Results:
| Exp | fps | mre | mae | rmse | rmse_log | log10 | delta_1 | delta_3 | delta_3 |
| :-- | :-- | :-- | :-- | :--- | :------- | :---- | :------ | :------ | :------ |
| HOHO_depthS_SGD_dct_efficienthc_TransEn1 | 106 | 0.1114 | 0.2197 | 0.4083 | 0.0737 | 0.0502 | 0.8671 | 0.9694 | 0.9916 |
| HOHO_depthS_dct_efficienthc_TransEn1 | 104 | 0.1040 | 0.2134 | 0.3940 | 0.0678 | 0.0475 | 0.8955 | 0.9749 | 0.9933 |


<br/>

### `Stanford2d3d` x `semantic segmentation`
Run:
```
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h64_fold1_simple.yaml
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h64_fold2_simple.yaml
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h64_fold3_simple.yaml

python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h256_fold1_simple.yaml
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h256_fold2_simple.yaml
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h256_fold3_simple.yaml

python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h1024_fold1_resnet101rgb.yaml
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h1024_fold2_resnet101rgb.yaml
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h1024_fold3_resnet101rgb.yaml

python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h1024_fold1_resnet101.yaml
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h1024_fold2_resnet101.yaml
python test_sem.py --cfg config/s2d3d_sem/HOHO_depth_dct_efficienthc_TransEn1_h1024_fold3_resnet101.yaml
```

Results:
| Exp | fps | iou | acc |
| :-- | :-- | :-- | :-- |
| HOHO_depth_dct_efficienthc_TransEn1_h64_fold1_simple | 202 | 43.04 | 53.06 |
| HOHO_depth_dct_efficienthc_TransEn1_h64_fold2_simple | 204 | 36.27 | 48.45 |
| HOHO_depth_dct_efficienthc_TransEn1_h64_fold3_simple | 202 | 43.14 | 54.81 |

| Exp | fps | iou | acc |
| :-- | :-- | :-- | :-- |
| HOHO_depth_dct_efficienthc_TransEn1_h256_fold1_simple | 135 | 46.49 | 56.33 |
| HOHO_depth_dct_efficienthc_TransEn1_h256_fold2_simple | 135 | 37.18 | 48.60 |
| HOHO_depth_dct_efficienthc_TransEn1_h256_fold3_simple | 135 | 46.09 | 56.81 |

| Exp | fps | iou | acc |
| :-- | :-- | :-- | :-- |
| HOHO_depth_dct_efficienthc_TransEn1_h1024_fold1_resnet101rgb | 10 | 53.94 | 64.30 |
| HOHO_depth_dct_efficienthc_TransEn1_h1024_fold2_resnet101rgb | 10 | 45.03 | 61.70 |
| HOHO_depth_dct_efficienthc_TransEn1_h1024_fold3_resnet101rgb | 10 | 56.87 | 68.94 |

| Exp | fps | iou | acc |
| :-- | :-- | :-- | :-- |
| HOHO_depth_dct_efficienthc_TransEn1_h1024_fold1_resnet101 | 10 | 59.05 | 68.91 |
| HOHO_depth_dct_efficienthc_TransEn1_h1024_fold2_resnet101 | 10 | 49.70 | 65.86 |
| HOHO_depth_dct_efficienthc_TransEn1_h1024_fold3_resnet101 | 10 | 60.28 | 71.85 |
