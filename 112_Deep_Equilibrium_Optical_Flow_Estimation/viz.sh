#!/bin/bash

python -u main.py --viz --name deq-flow-H-kitti  --stage kitti \
    --viz_set kitti --restore_ckpt checkpoints/deq-flow-H-kitti.pth --gpus 0 \
    --wnorm --f_thres 36 --f_solver broyden --huge 
