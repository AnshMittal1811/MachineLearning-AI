# Two-Stream Convolution Augmented Transformer for Human Activity Recognition

This repository contains the Pytorch implementation of the THAT methods in the following paper:

Two-Stream Convolution Augmented Transformer for Human Activity Recognition

[*Bing Li*](https://windofshadow.github.io/), Wei Cui, Wei Wang, [*Le Zhang*](https://zhangleuestc.github.io/), [*Zhenghua Chen*](https://zhenghuantu.github.io/) and [*Min Wu*](https://sites.google.com/site/wumincf/)

AAAI, 2021.

As illustrated in the following figure, THAT utilizes a two-stream structure to capture both time-over-channel and channel-over-time features, and use the multi-scale convolution augmented transformer to capture range-based patterns.

<div align=center><img src="https://github.com/windofshadow/THAT/blob/main/Architecture_new.jpg"  height="500" /></div>

## Requirements
- [python 3.7](We recommend to use Anaconda, since many python libs like numpy and sklearn are needed in our code.)
- [PyTorch 1.4.0](https://pytorch.org/) (we run the code under version 1.4.0, maybe versions >=1.0 also work.)  

## Dataset Downloads
Please Download the [data](https://github.com/ermongroup/Wifi_Activity_Recognition) and pre-process it as done in our paper.

## Training Example
CUDA_VISIBLE_DEVICES=0 python transformer-csi.py

## Notes
You may tune the hyperparameters to get further improved results.

## Citations
Please cite the following papers if you use this repository in your research work:
```sh


 @inproceedings{bing2021that,
  title={Two-Stream Convolution Augmented Transformer for Human Activity Recognition},
  author={Bing Li, Wei Cui, Wei Wang, Le Zhang, Zhenghua Chen and Min Wu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={},
  number={},
  year={2021}
}

```


Contact **Bing Li** [:envelope:](mailto:bing.li@unsw.edu.au) for questions, comments and reporting bugs.
 
