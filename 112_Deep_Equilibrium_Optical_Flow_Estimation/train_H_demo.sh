#!/bin/bash

python -u main.py --name deq-flow-H-1-step-grad-ad-things --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-H-chairs.pth \
    --gpus 0 1 --num_steps 120000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 \
    --wnorm --huge --f_solver anderson \
    --f_thres 60 --phantom_grad 1
