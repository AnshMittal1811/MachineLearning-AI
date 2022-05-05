#!/bin/bash

# Rethinking Portrait Matting with Privacy Preserving
# 
# Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
# Licensed under the MIT License (see LICENSE for details)
# Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting.git

result_dir=''
alpha_dir=''
trimap_dir=''

python core/eval.py \
--pred_dir $result_dir \
--alpha_dir $alpha_dir \
--trimap_dir $trimap_dir \