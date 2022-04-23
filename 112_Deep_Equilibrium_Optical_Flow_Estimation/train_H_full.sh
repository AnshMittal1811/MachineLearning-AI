#!/bin/bash


python -u main.py --name deq-flow-H-chairs --stage chairs --validation chairs \
    --gpus 0 1 2 --num_steps 120000 --batch_size 12 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
    --wnorm --huge --f_solver broyden \
    --f_thres 36 --n_losses 6 --phantom_grad 1 --sliced_core

python -u main.py --name deq-flow-H-things --stage things \
    --validation sintel kitti --restore_ckpt checkpoints/deq-flow-H-chairs.pth \
    --gpus 0 1 2 --num_steps 120000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 \
    --wnorm --huge --f_solver broyden \
    --f_thres 36 --n_losses 6 --phantom_grad 3 --sliced_core

python -u main.py --name deq-flow-H-sintel --stage sintel \
    --validation sintel --restore_ckpt checkpoints/deq-flow-H-things.pth \
    --gpus 0 1 2 --num_steps 120000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.0001 --gamma=0.90 \
    --wnorm --huge --f_solver broyden \
    --f_thres 36 --n_losses 6 --phantom_grad 3 --sliced_core

python -u main.py --name deq-flow-H-kitti --stage kitti \
    --validation kitti --restore_ckpt checkpoints/deq-flow-H-sintel.pth \
    --gpus 0 1 2 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.0001 --gamma=0.90 \
    --wnorm --huge --f_solver broyden \
    --f_thres 36 --n_losses 6 --phantom_grad 1 --sliced_core
