# PAMTRI: Pose-Aware Multi-Task Learning for Vehicle Re-Identification

This repo contains the official PyTorch implementation of *PAMTRI: Pose-Aware Multi-Task Learning for Vehicle Re-Identification Using Highly Randomized Synthetic Data*, ICCV 2019.

[[Paper](http://arxiv.org/abs/2005.00673)] [[Poster](figures/PAMTRI_poster.png)]

## Introduction

We address the problem of vehicle re-identification using multi-task learning and embeded pose representations. Since manually labeling images with detailed pose and attribute information is prohibitive, we train the network with a combination of real and randomized synthetic data. 

The proposed framework consists of two convolutional neural networks (CNNs), which are shown in the figure below. Top:  The pose estimation network is an extension of [high-resolution network (HRNet)](https://arxiv.org/abs/1902.09212) for predicting keypoint coordinates (with confidence/visibility) and generating heatmaps/segments. Bottom:  The multi-task network uses the embedded pose information from HRNet for joint vehicle re-identification and attribute classification. 

![Illustrating the architecture of PAMTRI](figures/pamtri.jpg)

## Getting Started

### Environment

The code was developed and tested with Python 3.6 on Ubuntu 16.04, using a NVIDIA GeForce RTX 2080 Ti GPU card. Other platforms or GPU card(s) may work but are not fully tested.

### Code Structure

Please refer to the `README.md` in each of the following directories for detailed instructions. 

- [PoseEstNet directory](PoseEstNet): The modified version of [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) for vehicle pose estimation. The code for training and testing, keypoint labels, and pre-trained models are provided. 

- [MultiTaskNet directory](MultiTaskNet): The multi-task network for joint vehicle re-identification and attribute classification using embedded pose representations. The code for training and testing, attribute labels, predicted keypoints, and pre-trained models are provided. 

## References

Please cite these papers if you use this code in your research:

    @inproceedings{Tang19PAMTRI,
      author = {Zheng Tang and Milind Naphade and Stan Birchfield and Jonathan Tremblay and William Hodge and Ratnesh Kumar and Shuo Wang and Xiaodong Yang},
      title = { {PAMTRI}: {P}ose-aware multi-task learning for vehicle re-identification using highly randomized synthetic data},
      booktitle = {Proc. of the International Conference on Computer Vision (ICCV)},
      pages = {211-–220},
      address = {Seoul, Korea},
      month = oct,
      year = 2019
    }

    @inproceedings{Tang19CityFlow,
      author = {Zheng Tang and Milind Naphade and Ming-Yu Liu and Xiaodong Yang and Stan Birchfield and Shuo Wang and Ratnesh Kumar and David Anastasiu and Jenq-Neng Hwang},
      title = {City{F}low: {A} city-scale benchmark for multi-target multi-camera vehicle tracking and re-identification},
      booktitle = {Proc. of the Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages = {8797–-8806},
      address = {Long Beach, CA, USA},
      month = jun,
      year = 2019
    }

## License

Code in the repository, unless otherwise specified, is licensed under the [NVIDIA Source Code License](LICENSE).

## Contact

For any questions please contact [Zheng (Thomas) Tang](https://github.com/zhengthomastang).
