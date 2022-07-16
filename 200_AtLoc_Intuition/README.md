[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
# [AtLoc: Attention Guided Camera Localization](https://arxiv.org/abs/1909.03557) - AAAI 2020 (Oral).

[Bing Wang](https://www.cs.ox.ac.uk/people/bing.wang/), [Changhao Chen](http://www.cs.ox.ac.uk/people/changhao.chen/website/), [Chris Xiaoxuan Lu](https://christopherlu.github.io/), [Peijun Zhao](https://www.cs.ox.ac.uk/people/peijun.zhao/), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), and [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/)

## License
Licensed under the CC BY-NC-SA 4.0 license, see [LICENSE](LICENSE.md).

## Introduction 

This is the PyTorch implementation of **AtLoc**, a simple and efficient neural architecture for robust visual localization.

#### Demos and Qualitative Results (click below for the video)
<p align="center"> <a href="https://youtu.be/_8NQXBadklU"><img src="./figures/real.gif" width="100%"></a> </p>

## Setup

AtLoc uses a Conda environment that makes it easy to install all dependencies.

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 2.7.

2. Create the `AtLoc` Conda environment: `conda env create -f environment.yml`.

3. Activate the environment: `conda activate py27pt04`.

4. Note that our code has been tested with PyTorch v0.4.1 (the environment.yml file should take care of installing the appropriate version).

## Data
We support the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and [Oxford RobotCar](http://robotcar-dataset.robots.ox.ac.uk/) datasets right now. You can also write your own PyTorch dataloader for other datasets and put it in the `data` directory.

### Special instructions for RobotCar:

1. Download [this fork](https://github.com/samarth-robo/robotcar-dataset-sdk/tree/master) of the dataset SDK, and run `cd data && ./robotcar_symlinks.sh` after editing the `ROBOTCAR_SDK_ROOT` variable in it appropriately.

2. For each sequence, you need to download the `stereo_centre`, `vo` and `gps` tar files from the dataset website. The directory for each 'scene' (e.g. `loop`) has .txt files defining the train/test_split.

3. To make training faster, we pre-processed the images using `data/process_robotcar.py`. This script undistorts the images using the camera models provided by the dataset, and scales them such that the shortest side is 256 pixels.

4. Pixel and Pose statistics must be calculated before any training. Use the `data/dataset_mean.py`, which also saves the information at the proper location. We provide pre-computed values for RobotCar and 7Scenes.

## Running the code

### Training
The executable script is `train.py`. For example:

- AtLoc on `loop` from `RobotCar`: 
```
python train.py --dataset RobotCar --scene loop --model AtLoc --gpus 0
```

- AtLocLstm on `loop` from `RobotCar`: 
```
python train.py --dataset RobotCar --scene loop --model AtLoc --lstm True --gpus 0
```

- AtLoc+ on `loop` from `RobotCar`: 
```
python train.py --dataset RobotCar --scene loop --model AtLocPlus --gamma -3.0 --gpus 0
```

The meanings of various command-line parameters are documented in train.py. The values of various hyperparameters are defined in `tools/options.py`.

### Inference
The trained models for partial experiments presented in the paper can be downloaded [here](https://drive.google.com/drive/folders/1inY29zupeCmvIF5SsJhQDEzo_jzY0j6Q). The inference script is `eval.py`. Here are some examples, assuming the models are downloaded in `logs`.

- AtLoc on `loop` from `RobotCar`: 
```
python eval.py --dataset RobotCar --scene loop --model AtLoc --gpus 0 --weights ./logs/RobotCar_loop_AtLoc_False/models/epoch_300.pth.tar
```

Calculates the network attention visualizations and saves them in a video

- For the AtLoc model trained on `loop` in `RobotCar`:
```
python saliency_map.py --dataset RobotCar --scene loop --model AtLoc --gpus 0 --weights ./logs/RobotCar_loop_AtLoc_False/models/epoch_300.pth.tar 
```

## Citation
If you find this code useful for your research, please cite our paper

```
@article{wang2019atloc,
  title={AtLoc: Attention Guided Camera Localization},
  author={Wang, Bing and Chen, Changhao and Lu, Chris Xiaoxuan and Zhao, Peijun and Trigoni, Niki and Markham, Andrew},
  journal={arXiv preprint arXiv:1909.03557},
  year={2019}
}
```
## Acknowledgements
Our code partially builds on [MapNet](https://github.com/NVlabs/geomapnet) and [PoseLstm](https://github.com/hazirbas/poselstm-pytorch)
