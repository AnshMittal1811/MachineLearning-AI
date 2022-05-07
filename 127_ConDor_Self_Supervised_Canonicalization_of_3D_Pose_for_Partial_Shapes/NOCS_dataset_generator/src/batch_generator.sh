#!/bin/bash

# Request a GPU partition node and access to 1 GPU

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 16
#SBATCH --mem=60G

#SBATCH -t 48:00:00
#SBATCH -o unocs_%j.out

# Load CUDA module
module load gcc/8.3
module load cuda/10.2

# Compile CUDA program and run
source ~/anaconda3/bin/activate
conda activate unocs_3d


python3 main.py --input /gpfs/scratch/rsajnani/rsajnani/research/dataset/shapenetcoco_v1/val/02691156 --output ./val_02691156.h5 --num_points 1024 --proc 16 --max_folders 100
echo "done"
