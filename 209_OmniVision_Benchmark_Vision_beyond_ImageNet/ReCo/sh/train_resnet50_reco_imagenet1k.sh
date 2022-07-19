#!/bin/sh

currenttime=`date "+%Y%m%d_%H%M%S"`
jobname=$1
gpu_num=1

mkdir -p logs
LOG_FILE=logs/train-log-$jobname-$currenttime

python ../reco_imagenet.py \
    --arch resnet50 \
    --alpha 0.05 \
    --beta 1.0 \
    --wd 1e-4 \
    --gamma 1.0 \
    --mark Reco_r50_imagenet1k \
    --lr 0.64 \
    -b 128 \
    --moco-t 0.2 \
    --moco-k 8192 \
    --aug randcls_sim \
    --epochs 200 \
    2>&1 | tee $LOG_FILE > /dev/null &

echo -e "\033[32m[ Please see LOG_FILE for details: \"tail -f ${LOG_FILE}\" ]\033[0m"