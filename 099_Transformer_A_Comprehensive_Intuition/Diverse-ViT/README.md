# The Principle of Diversity: Training Stronger Vision Transformers Calls for Reducing All Levels of Redundancy

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Codes for this paper: [CVPR 2022] [The Principle of Diversity: Training Stronger Vision Transformers Calls for Reducing All Levels of Redundancy](). 

Tianlong Chen, Zhenyu Zhang, Yu Cheng, Ahmed Awadallah, Zhangyang Wang.



## Overview

Vision transformers (ViTs) have gained increasing popularity as they are commonly believed to own higher modeling capacity and representation flexibility, than traditional convolutional networks. However, it is questionable whether such potential has been fully unleashed in practice, as the learned ViTs often suffer from over-smoothening, yielding likely redundant models. 

Recent works made preliminary attempts to identify and alleviate such redundancy, e.g., via regularizing embedding similarity or re-injecting convolution-like structures. However, a “head-to-toe assessment” regarding the extent of redundancy in ViTs, and how much we could gain by thoroughly mitigating such, has been absent for this field. 

This paper, for the first time, systematically studies the ubiquitous existence of redundancy at all three levels: patch embedding, attention map, and weight space. In view of them, we advocate a principle of diversity for training ViTs, by presenting corresponding regularizers that encourage the representation diversity and coverage at each of those levels, that enabling capturing more discriminative information. 

Extensive experiments on ImageNet with a number of ViT backbones validate the effectiveness of our proposals, largely eliminating the observed ViT redundancy and significantly boosting the model generalization. For example, our diversified DeiT obtains 0.70% ∼1.76% accuracy boosts on ImageNet with highly reduced similarity.

<img src = "Figs/Diversity_overview.png" align = "center" width="100%" hight="60%">



## Prerequisites

Install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch torchvision
pip install timm==0.3.2
```



## Training on ImageNet

```
./script/run_deit_small_diverse.sh [data/imagenet] (Deit-Small-12layers)
./script/run_deit_small_24layer_diverse.sh [data/imagenet] (Deit-Small-24layers)
```



## Citation

```
TBD
```



## Acknowledgement

https://github.com/facebookresearch/deit
