#!/bin/bash
python $2 --config_file $3 --gpus $4 --train_file $2 2>&1 | tee logs/$1.txt

# bash ./train_pc.sh cdc_x4 ./CDC_train_test.py ./options/realSR_HGSR_MSHR.py 1
