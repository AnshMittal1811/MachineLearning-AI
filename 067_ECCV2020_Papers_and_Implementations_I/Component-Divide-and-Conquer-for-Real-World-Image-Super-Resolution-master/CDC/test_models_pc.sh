#!/bin/bash
python $2 --result_dir $1 --pretrain $3 --gpus $4 2>&1 | tee logs/TestLogs/$1.txt

# bash test_models_pc.sh cdc_x4_test ./CDC_test.py ./models/HGSR-MHR_X4_SubRegion_GW_283.pth 1
