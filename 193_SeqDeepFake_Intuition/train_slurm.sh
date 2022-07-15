LOGNAME=log$(date +"%Y%m%d_%H%M%S")
CONFIG='./configs/r50.json'
DATA_DIR='YOUR-DATASET-ROOT-HERE'
DATASET_NAME='facial_attributes'
RESULTS_DIR='./results'

HOST='127.0.0.1'
PORT='12345'
NUM_GPU=4

PARTITION='YOUR-PARTITION-HERE'
NODE='YOUR-NODE-HERE'

srun -p ${PARTITION} --mpi=pmi2 --gres=gpu:$NUM_GPU --ntasks-per-node=${NUM_GPU} -n1 -w ${NODE}\
    --job-name=seqdeepfake --kill-on-bad-exit=1 --cpus-per-task=4 \
    python train.py \
        --log_name ${LOGNAME} \
        --cfg ${CONFIG} \
        --data_dir ${DATA_DIR} \
        --dataset_name ${DATASET_NAME} \
        --val_epoch 10 \
        --model_save_epoch 5 \
        --manual_seed 777 \
        --dist-url tcp://${HOST}:${PORT} \
        --world_size ${NUM_GPU} \
        --rank 0 \
        --launcher pytorch \
        --results_dir ${RESULTS_DIR}
