# CaDeX

### CVPR2022 [Jiahui Lei](https://www.cis.upenn.edu/~leijh/) and [Kostas Daniilidis](https://www.cis.upenn.edu/~kostas/); [Project Page](https://www.cis.upenn.edu/~leijh/projects/cadex/)

### CaDeX: Learning mayonical Deformation Coordinate Space for Dynamic Surface Representation via Neural Homeomorphism

![image](./assets/idea.png)

## Install

Install the environment following `scripts/install_env.sh`, which is for cuda 11.0 and pytorch 1.7.1. This repo should also work with cuda 10.x, but you need to change the cuda version in the script. To install with cuda 11.0, use the command `bash scripts/install_env.sh`

## Prepare the data and Pre-trained weights

**Download**:

- By default, you may download processed D-FAUST data by [this script](https://github.com/autonomousvision/occupancy_flow/blob/master/scripts/download_data.sh) from Occupancy Flow. The current configuration is for this dataset.
- You may download all the data we preprocessed from google drive [here](https://drive.google.com/drive/folders/1ROekQKWMoTxP655RqU6eTBf0dbtanweC?usp=sharing). We provide:
  - `dt4d_v3.zip`, preprocessed DeformingThings4D animal bodies.
  - `OBJECTNAME_cadex.zip`, preprocessed Shape2Motion articulated objects. The `OBJECTNAME` can be `laptop`,`door`, `stapler`, `washing_machine`, `oven`, `eyeglasses`, `refrigerator`.
  - `Humans_multi.zip`, As an alternative to O-Flow's processed D-FAUST data, we provide its multi-file slicing to boost the dataloader speed on the network file system (e.g. a large gpu cluster). You don't need to download this if your disk IO is not a bottleneck, and you may use the dataset from O-Flow directly. To use this multi-file dataset, you need to modify the config yaml file, see details [bellow](#Humans_multi.zip).
- You may download the pre-trained model weights from google drive [here](https://drive.google.com/drive/folders/1yAhnL9eufI7hAXUGG8CsOQ8IUXAqDCxr?usp=sharing)

You may find useful [tool](https://github.com/prasmussen/gdrive) to download from google drive by linux command line.

**Unzip and put to proper places**:

After you download the dataset and pre-trained models you are interested in (for example, you don't need to download dt4d_v3.zip if you are only interested in human bodies experiments), make or link directories under project root  `mkdir -p resource/data; mkdir -p resource/checkpoints` and unzip the dataset, the proper organization would be the following:

```shell
./resource
├── checkpoints
│   ├── dfaust
│   │   ├── dfaust_w_pf.pt
│   │   └── ...
│   ├── dt4d
│   │   ├── dt4d_dep.pt
│   │   └── dt4d_pcl.pt
│   └── s2m
│       ├── s2m_pcl_door.pt
│       ├── s2m_dep_eyeglasses.pt
│       └── ...
└── data
    ├── dt4d_v3 # For Animals
    │   ├── bear3EP_Agression
    │   └── ...
    ├── Humans # Or Humans_multi
    │   └── D-FAUST
    └── s2m # For Articulated Objects
        ├── door
        ├── stapler
        └── ...
```

## Evaluate with pre-trained models

Before running any training or testing, make sure you have a log directory (or link) under the project root by `mkdir log`

We provide all configuration files for three categories, you can check them under `configs` directory. An example of running a testing is:
```shell
# under the project root

python run.py --config ./configs/dfaust/testing/dfaust_w_pf_test_seen.yaml -f 

# -f is to bypass the interactive confirmation, otherwise you need to interactively confirm the config and the running will start
# to see other useful flags, you may find it under /init/pre_config.py
```
After the evaluation is finished, you will find the corresponding log sub-folder under the `log` directory. Under each sub-folder, there will be an `xls` subfolder, and the evaluation report will be there. The log for each experiment also includes the tensorboard log and visualization.

## Train CaDeX

The training configs are provided in `configs` as well. Our default training setup is 2x2080ti GPUs, An example to run the training is:
```shell
python run.py --config ./configs/dfaust/training/dfaust_w_pf.yaml -f 
```

## Humans_multi.zip
If you want to use our processed data for D-FAUST human bodies with multi-file slicing that boosts the disk IO, you have to change two configurations in the config yaml file:
- `path: resource/data/Humans` should be changed to somewhere you unzipped the `Humans_multi.zip`, e.g: `path: resource/data/Humans_multi`
- `training_multi_files: false` should be changed to `training_multi_files: true`

## TODO

- clean the code

## Citation

If you find this repo useful, please consider to cite our paper, thanks!
```latex
@inproceedings{Lei2022CaDeX,
title={CaDeX: Learning Canonical Deformation Coordinate Space for Dynamic Surface Representation via Neural Homeomorphism},
author={Lei, Jiahui and Daniilidis, Kostas},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
url={https://cis.upenn.edu/~leijh/projects/cadex},
year={2022}
} 
```

## Reference

This project is based on several wonderful projects:

- OccNet:  <https://github.com/autonomousvision/occupancy_networks>
- O-FLow:  <https://github.com/autonomousvision/occupancy_flow>
- Neural Parts: <https://github.com/paschalidoud/neural_parts>
- DeformingThings4D: <https://github.com/rabbityl/DeformingThings4D>
- LPDC-Net: <https://github.com/Gorilla-Lab-SCUT/LPDC-Net>
- A-SDF: <https://github.com/JitengMu/A-SDF>
