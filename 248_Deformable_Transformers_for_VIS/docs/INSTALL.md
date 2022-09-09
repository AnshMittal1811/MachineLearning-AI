# Installation

1. Clone and enter this repository:
    ```
    git clone git@github.com:acaelles97/DeVIS.git
    cd DeVIS
    ```
2. Install packages for Python 3.8:
   1. Install PyTorch 1.11.0 and torchvision 0.12.0 from [here](https://pytorch.org/get-started/locally/). The tested CUDA version is 11.3.0 
   2. `pip3 install -r requirements.txt`
   3. Install [youtube-vis](https://github.com/youtubevos/cocoapi) api 
   ```
   pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI
   ```
   4. Install MultiScaleDeformableAttention package:
    ```
   cd src/models/ops/
   python setup.py build_ext install
   ```

Check [this](https://github.com/Epiphqny/VisTR/issues/5) if you experience problems installing youtube-vis api


## Dataset preparation
First step is to download and extract each dataset: [COCO](https://cocodataset.org/#home), [YT-19](https://youtube-vos.org/dataset/vis/), [YT-21](https://youtube-vos.org/dataset/vis/) & [OVIS](http://songbai.site/ovis/)
User must set `DATASETS.DATA_PATH` to the root data path. 
We refer to [`src/datasets/coco.py`](../src/datasets/coco.py) & [`src/datasets/vis.py`](../src/datasets/vis.py) to modify the expected format for COCO and VIS datasets respectively.
We expect the following organization:
```
cfg.DATASETS.DATA_PATH/
└── COCO/
  ├── train2017/
  ├── val2017/
  └── annotations/
      ├── instances_train2017.json
      └── instances_val2017.json
 
└── Youtube_VIS-2019/
  ├── train/
      ├── JPEGImages
      └── train.json 
  └── valid/
      ├── JPEGImages
      └── valid.json 

└── Youtube_VIS-2021/
  ├── train/
      ├── JPEGImages
      └── instances.json 
  └── valid/
      ├── JPEGImages
      └── instances.json

└── OVIS/
  ├── train/
  ├── annotations_train.json/
  ├── valid/     
  └── annotations_valid.json/

```

## Download pre-trained weights
We provide pre-trained weights for the Deformable Mask Head training, as well as DeVIS (including ablations). 
We expect them to be downloaded and unpacked under the [`weights`](../weights) directory
```
cd weights
wget https://vision.in.tum.de/webshare/u/cad/ablation_pre-trained_weights.zip
unzip ablation_pre-trained_weights.zip
```
