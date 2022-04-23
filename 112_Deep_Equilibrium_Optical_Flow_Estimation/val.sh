#!/bin/bash

python -u main.py --eval --name deq-flow-B-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-B-things-test.pth --gpus 0 \
    --wnorm --f_thres 40 --f_solver anderson 

python -u main.py --eval --name deq-flow-H-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-H-things-test-1.pth --gpus 0 \
    --wnorm --f_thres 36 --f_solver broyden --huge 

python -u main.py --eval --name deq-flow-H-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-H-things-test-2.pth --gpus 0 \
    --wnorm --f_thres 36 --f_solver broyden --huge 

python -u main.py --eval --name deq-flow-H-things  --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-H-things-test-3.pth --gpus 0 \
    --wnorm --f_thres 36 --f_solver broyden --huge 
