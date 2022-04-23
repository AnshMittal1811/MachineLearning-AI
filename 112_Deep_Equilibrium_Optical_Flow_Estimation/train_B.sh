#!/bin/bash

python -u main.py --name deq-flow-B-chairs --stage chairs --validation chairs \
    --gpus 0 1 --num_steps 120000 --batch_size 12 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
    --wnorm --f_solver anderson --f_thres 36 \
    --n_losses 6 --phantom_grad 1

python -u main.py --name deq-flow-B-things --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-B-chairs.pth \
    --gpus 0 1 --num_steps 120000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 \
    --wnorm --f_solver anderson --f_thres 40 \
    --n_losses 2 --phantom_grad 3

