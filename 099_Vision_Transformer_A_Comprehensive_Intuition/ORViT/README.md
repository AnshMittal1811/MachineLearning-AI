# ORViT: Object-Region Video Transformers

This is an official pytorch implementation of paper [Object-Region Video Transformers](https://arxiv.org/abs/2110.06915). In this repository, we provide the PyTorch code we used to train and test our proposed ORViT layer.

If you find ORViT useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@misc{orvit2021,
      author={Roei Herzig and Elad Ben-Avraham and Karttikeya Mangalam and Amir Bar and Gal Chechik and Anna Rohrbach and Trevor Darrell and Amir Globerson},
      title={Object-Region Video Transformers},
      year={2021},
      eprint={2110.06915},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Model Zoo

| name | dataset | # of frames | spatial crop | acc@1 | acc@5 | url |
| --- | --- | --- | --- | --- | --- | --- |
| ORViT-Motionformer | SSv2 | 16 | 224 | 67.9 | 90.8 | [model](https://drive.google.com/file/d/1hDyPwBnif0ud3hQY8615bIzyR5vH5uJk/view?usp=sharing) |
| ORViT-Motionformer-HR | EK100 | 16 | 336 | 45.7 | 75.8 | [model](https://drive.google.com/file/d/13PSMc-iboyt2S_w-sRXAZduzknLmvZ8j/view?usp=sharing) |



# Installation

First, create a conda virtual environment and activate it:
```
conda create -n orvit python=3.8.5 -y
source activate orvit
```

Then, install the following packages:

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- matplotlib: `pip install matplotlib`
- pandas: `pip install pandas`
- ffmeg: `pip install ffmpeg-python`

OR:

simply create conda environment with all packages just from yaml file:

`conda env create -f environment.yml`

Lastly, build the ORViT codebase by running:
```
git clone https://github.com/eladb3/ORViT.git
cd ORViT
python setup.py build develop
```

# Usage

## Dataset Preparation

Please use the dataset preparation instructions provided in [DATASET.md](slowfast/datasets/DATASET.md).

Boxes for SSv2 and splits for Something-Else datasets can be downloaded from https://github.com/joaanna/something_else.

## Training the ORViT~MF

Training the default ORViT that uses Motionformer as backbone, and operates on 16-frame clips cropped at 224x224 spatial resolution, can be done using the following command:

```
python tools/run_net.py \
  --cfg configs/K400/motionformer_224_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`, or you can simply modify

```
DATA:
  PATH_TO_DATA_DIR: path_to_your_dataset
```

To the yaml configs file, then you do not need to pass it to the command line every time.


## Inference

Use `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for a given run. When testing, you also have to provide the path to the checkpoint model via TEST.CHECKPOINT_FILE_PATH.
```
python tools/run_net.py \
  --cfg configs/K400/motionformer_224_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```


# Acknowledgements

ORViT is built on top of [PySlowFast](https://github.com/facebookresearch/SlowFast), [PySlowFast](https://github.com/facebookresearch/Motionformer), [Motionformer](https://github.com/facebookresearch/TimeSformer) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman). We thank the authors for releasing their code. We thank Dantong Niu for helping us test this repository. If you use our model, please consider citing these works as well:

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```

```BibTeX
@misc{patrick2021keeping,
      title={Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers}, 
      author={Mandela Patrick and Dylan Campbell and Yuki M. Asano and Ishan Misra Florian Metze and Christoph Feichtenhofer and Andrea Vedaldi and Jo\ão F. Henriques},
      year={2021},
      eprint={2106.05392},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```BibTeX
@inproceedings{gberta_2021_ICML,
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle   = {Proceedings of the International Conference on Machine Learning (ICML)}, 
    month = {July},
    year = {2021}
}
```

```BibTeX
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
