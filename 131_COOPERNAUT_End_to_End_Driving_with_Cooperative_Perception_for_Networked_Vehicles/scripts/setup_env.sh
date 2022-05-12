#!/bin/bash
# on TACC you need to load these module to successfully install autocast environment
module load gcc
module load cuda/10.1
module load python3
#create environment
conda env create -f environment.yml
conda activate autocast
conda install -c hargup/label/pypi mosquitto

