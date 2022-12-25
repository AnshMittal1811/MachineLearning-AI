# PTSEFormer

This repo is an official implementation of PTSEFormer, which is accepted by ECCV 2022. 

## Citing PTSEFormer

Please consider citing our paper if you find it useful:

```
@article{ptseformer,
    Author = {Han, Wang and Jun, Tang and Xiaodong, Liu and Shanyan, Guan and Rong, Xie and Li, Song},
    Title = {PTSEFormer: Progressive Temporal-Spatial Enhanced TransFormer Towards Video Object Detection},
    Conference = {ECCV},
    Year = {2022}
}
```

## Installation

### Requirements

- Linux, CUDA>=9.2, GCC>=5.4

- Python>=3.7

  ```
  conda create -n PTSEFormer python=3.7 pip
  ```

  Then, activate the environment:

  ```
  conda activate PTSEFormer
  ```

- PyTorch>=1.5.1, torchvision>=0.6.1 

  ```
  conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
  ```

- Other requirements

  ```
  pip install -r requirements.txt
  ```

## Data preparation

Please download the ILSVRC2015 DET and ILSVRC2015 VID dataset from [here](http://image-net.org/challenges/LSVRC/2015/2015-downloads) and organize them as following. 

    data_root/
    	└──ILSVRC2015/
    		├──ImageSets/
    		├──Annotations/
    		├──Data/

### Evaluation

The inference command line for testing on the validation dataset:

    python -m torch.distributed.launch --nproc_per_node=8 tools/test.py --config-file experiments/PTSEFormer_r101_8gpus.yaml

### Training

The training command line for training on a combined dataset of VID and DET:

    python -m torch.distributed.launch --nproc_per_node=8 tools/train.py --config-file experiments/PTSEFormer_r101_8gpus.yaml

