#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 2
#SBATCH --mem=16G

#SBATCH -t 48:00:00
#SBATCH -o unocs_%j.out

# Load CUDA module
#module load gcc/8.3
#module load cuda/10.2

# Compile CUDA program and run
source ~/anaconda3/bin/activate
conda activate TFN_capsules

#cd ../
CUDA_VISIBLE_DEVICES=0 python3 train_pointnet_cls2.py

