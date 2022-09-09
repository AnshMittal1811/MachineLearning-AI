# HQ-YTVIS data loading and evaluation

It support both the tube-boundary AP evalution proposed in Video Mask Transfiner for High-Quality Video Instance Segmentation [ECCV 2022], and also the traditional tube-mask AP evaluation.
## Introduction

This package provides data loading and evaluation functionalities for high-quality video instance segmentation on HQ-YTVIS. It is built based on [youtubevos API](https://github.com/youtubevos/cocoapi/) designed for the Youtube VOS dataset (https://youtube-vos.org/dataset/vis/). For evaluation metrics, pleae refer to the Video Mask Transfiner for High-Quality Video Instance Segmentation [ECCV 2022].

We have only implemented Python API for HQ-YTVIS.

## Installation
To install:
```
cd PythonAPI
# To compile and install locally 
python setup.py build_ext --inplace
# To install library to Python site-packages 
python setup.py build_ext install
```

## Contact
If you have any questions regarding the repo, please create an issue.
