#!/bin/bash

# Rethinking Portrait Matting with Privacy Preserving
# 
# Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
# Licensed under the MIT License (see LICENSE for details)
# Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting.git
# Paper link: https://arxiv.org/abs/2203.16828

arch='vitae'
model_path='./models/P3M-Net_ViTAE-S_trained_on_P3M-10k.pth'
dataset='SAMPLES'
test_choice='HYBRID'

python core/infer.py \
--arch $arch \
--dataset $dataset \
--model_path $model_path \
--test_choice $test_choice \