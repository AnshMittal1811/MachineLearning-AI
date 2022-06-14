# High-Resolution Network (HRNet) for Vehicle Pose Estimation

## Introduction

This directory is an extension of the [PyTorch implementation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) of [high-resolution network (HRNet)](https://arxiv.org/abs/1902.09212) for vehicle pose estimation. 

The original framework was used for the **person-based** pose estimation problem with a focus on learning reliable high-resolution representations. In this work, we are interested in the **vehicle-based** problem, where the targets are rigid bodies with high intra-class variability. The original code has been modified for 36-keypoint-based vehicle pose estimation.

## Getting Started

### Installation

We highly recommend to create a virtual environment for the following steps. For example, an introduction to Conda environments can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 

1. Install PyTorch >= v1.0.0 following the [instructions](https://pytorch.org/).
2. Clone the repo, and change the current working directory to `PoseEstNet`, which will be referred to as `${POSE_ROOT}`:
   ```
   cd ${POSE_ROOT}
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download models: 
   ```
   wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1vD08fh-za3mgTJ9UkK1ASCTJAqypW0RL' -O models.zip
   unzip models.zip
   rm models.zip
   ```
5. Make libs:
   ```
   cd lib
   make
   ```
6. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   The line `# COCOAPI=/path/to/install/cocoapi` indicates that you need to specify a path to have the repo cloned and then set an environment variable (`COCOAPI` in this case) accordingly.
7. Change the current working directory back to `${POSE_ROOT}`, and create directories for output and log:
   ```
   mkdir output 
   mkdir log
   ```
   Your directory tree should look like this:
   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── LICENSE
   ├── README.md
   └── requirements.txt
   ```

### Datasets

- [VeRi](https://vehiclereid.github.io/VeRi/)
- [CityFlow-ReID](https://www.aicitychallenge.org/2020-data-and-evaluation/) (Equivalent to the Track 2 data for the 4th AI City Challenge Workshop at CVPR 2020)
- At this time, we are not able to release the synthetic data used in our [paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Tang_PAMTRI_Pose-Aware_Multi-Task_Learning_for_Vehicle_Re-Identification_Using_Highly_Randomized_ICCV_2019_paper.html). However, you may use the [NVIDIA Deep learning Dataset Synthesizer (NDDS)](https://github.com/NVIDIA/Dataset_Synthesizer), or another equivalent tool, to generate your own synthetic data with pose information. 

We adopt the 36-keypoint vehicle model as used by other researchers<sup>[1](https://arxiv.org/abs/1803.02057),[2](https://arxiv.org/abs/1411.5935)</sup> and define 13 segments of vehicle surface accordingly. 

![Definition of 36 keypoints and 13 segments](../figures/keypoints_segments.png)

Please prepare the `data` directory according to the following format.
   ```
   ${POSE_ROOT}
    `-- data
        |-- veri
            |-- images
            |   |-- image_test
            |   |   |-- 0002_c002_00030600_0.jpg
            |   |   |-- 0002_c002_00030605_1.jpg
            |   |   |-- ...
            |   `-- image_train
            |   |   |-- 0001_c001_00016450_0.jpg
            |   |   |-- 0001_c001_00016460_0.jpg
            |   |   |-- ...
            `-- annot
            |   |-- label_test.csv
            |   `-- label_train.csv
        |-- ...

   ```

In the `label_*.csv` files, the format of each line is defined as follows. The visibility/confidence is a binary value with 1 for a visible keypoint and 0 for an invisible one. 
```
<image name>,<image width>,<image height>,<x of keypoint0>,<y of keypoint0>,<visibility of keypoint0>,<x of keypoint1>,<y of keypoint1>,<visibility of keypoint1>,...,<x of keypoint35>,<y of keypoint35>,<visibility of keypoint35>
```

An example is shown below. 
```
0001_c001_00016450_0.jpg,126,135,121,55,0,118,95,0,117,111,1,118,87,1,120,76,1,118,97,1,119,66,1,123,46,1,121,38,1,121,54,1,115,47,1,107,12,1,109,15,1,99,9,1,97,24,1,109,56,1,107,79,1,111,115,1,27,63,0,13,103,0,12,120,1,9,94,1,13,86,0,17,105,0,26,75,0,22,55,0,27,46,0,30,62,0,37,52,0,41,12,0,37,15,0,38,10,1,31,28,1,19,62,1,18,87,1,16,124,1
```

Our labels for a subset of VeRi are provided with the repo. 

### Training and Testing

#### Testing

Run the following command to test an inference model, where the configuration parameters are defined in the YAML file at `experiments/<dataset>/<network>/<config>`.
```
python tools/test.py --evaluate --outputPreds --outputDir output --cfg experiments/<dataset>/<network>/<config> TEST.MODEL_FILE models/<dataset>/<network>/<config>/<weights>
```

Some examples are given as follows:
```
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/veri/hrnet/w32_256x256_adam_lr1e-3.yaml TEST.MODEL_FILE models/veri/pose_hrnet/w32_256x256_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/cityflow/hrnet/w32_256x256_adam_lr1e-3.yaml TEST.MODEL_FILE models/cityflow/pose_hrnet/w32_256x256_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/synthetic/hrnet/w32_256x256_adam_lr1e-3.yaml TEST.MODEL_FILE models/synthetic/pose_hrnet/w32_256x256_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/veri_synthetic/hrnet/w32_256x256_adam_lr1e-3.yaml TEST.MODEL_FILE models/veri_synthetic/pose_hrnet/w32_256x256_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/cityflow_synthetic/hrnet/w32_256x256_adam_lr1e-3.yaml TEST.MODEL_FILE models/cityflow_synthetic/pose_hrnet/w32_256x256_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/veri/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/veri/pose_resnet/res50_256x256_d256x3_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/cityflow/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/cityflow/pose_resnet/res50_256x256_d256x3_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/synthetic/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/synthetic/pose_resnet/res50_256x256_d256x3_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/veri_synthetic/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/veri_synthetic/pose_resnet/res50_256x256_d256x3_adam_lr1e-3/model_best.pth
python tools/test.py --outputPreds --evaluate --outputDir output --cfg experiments/cityflow_synthetic/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/cityflow_synthetic/pose_resnet/res50_256x256_d256x3_adam_lr1e-3/model_best.pth
```

The argument `--evaluate` is for evaluation of the given model, which can be disabled when using the model for inference only (The test set of images can be changed in the configuration YAML file). Note that the input CSV file of labels is also necessary in inference mode, but entries for joints will not be used for evaluation - Feel free to put dumb values as placeholders. The argument `--outputPreds` enables the output of predicted pose (keypoints), heatmaps and segments for pose-aware re-identification, where the output directory is specified by `--outputDir output`. In the output CSV file of pose (keypoints), the format of each line is defined as follows. 
```
<image name>,<x of keypoint0>,<y of keypoint0>,<visibility of keypoint0>,<x of keypoint1>,<y of keypoint1>,<visibility of keypoint1>,...,<x of keypoint35>,<y of keypoint35>,<visibility of keypoint35>
```

#### Training

Run the following command to start training process, where the configuration parameters are defined in the YAML file at `experiments/<dataset>/<network>/<config>`.
```
python tools/train.py --cfg experiments/<dataset>/<network>/<config>
```

Some examples are given as follows:
```
python tools/train.py --cfg experiments/veri/hrnet/w32_256x256_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/cityflow/hrnet/w32_256x256_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/veri_synthetic/hrnet/w32_256x256_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/cityflow_synthetic/hrnet/w32_256x256_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/synthetic/hrnet/w32_256x256_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/veri/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/cityflow/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/veri_synthetic/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/cityflow_synthetic/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml
python tools/train.py --cfg experiments/synthetic/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml
```

Note: If `tensorboardX.SummaryWriter` is causing errors, you may remove all the related usage in [`tools/train.py`](tools/train.py). 

## References

Please cite these papers if you use this code in your research:

    @inproceedings{Tang19PAMTRI,
      author = {Zheng Tang and Milind Naphade and Stan Birchfield and Jonathan Tremblay and William Hodge and Ratnesh Kumar and Shuo Wang and Xiaodong Yang},
      title = { {PAMTRI}: {P}ose-aware multi-task learning for vehicle re-identification using highly randomized synthetic data},
      booktitle = {Proc. of the International Conference on Computer Vision (ICCV)},
      pages = {211-–220},
      address = {Seoul, Korea},
      month = Oct,
      year = 2019
    }

    @inproceedings{Tang19CityFlow,
      author = {Zheng Tang and Milind Naphade and Ming-Yu Liu and Xiaodong Yang and Stan Birchfield and Shuo Wang and Ratnesh Kumar and David Anastasiu and Jenq-Neng Hwang},
      title = {City{F}low: {A} city-scale benchmark for multi-target multi-camera vehicle tracking and re-identification},
      booktitle = {Proc. of the Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages = {8797–-8806},
      address = {Long Beach, CA, USA},
      month = Jun,
      year = 2019
    }

    @inproceedings{Sun19HRNet,
      author = {Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
      title = {Deep high-resolution representation learning for human pose estimation},
      booktitle = {Proc. of the Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages = {5693--5703},
      address = {Long Beach, CA, USA},
      month = Jun,
      year = 2019
    }

## License

Code in this directory is licensed under the [MIT License](LICENSE).

## Contact

For any questions please contact [Zheng (Thomas) Tang](https://github.com/zhengthomastang).
