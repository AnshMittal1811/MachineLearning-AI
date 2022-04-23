#!/bin/bash

python -u main.py --name deq-flow-B-1-step-grad-things --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-B-chairs.pth \
    --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 \
    --wnorm --f_thres 40 --f_solver anderson \
