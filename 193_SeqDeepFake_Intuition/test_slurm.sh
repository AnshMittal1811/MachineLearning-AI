LOGNAME='YOUR-LOGNAME-HERE'
CONFIG='./configs/r50.json'
DATA_DIR='YOUR-DATASET-ROOT-HERE'
DATASET_NAME='facial_attributes'
TEST_TYPE='adaptive'
RESULTS_DIR='./results'

HOST='127.0.0.1'
PORT='12345'
NUM_GPU=1

PARTITION='YOUR-PARTITION-HERE'
NODE='YOUR-NODE-HERE'

srun -p ${PARTITION} --mpi=pmi2 --gres=gpu:${NUM_GPU} --ntasks-per-node=${NUM_GPU} -n1 -w ${NODE}\
    --job-name=test --kill-on-bad-exit=1 --cpus-per-task=1 \
    python test.py \
        --log_name ${LOGNAME} \
        --cfg ${CONFIG} \
        --data_dir ${DATA_DIR} \
        --dataset_name ${DATASET_NAME} \
        --test_type ${TEST_TYPE} \
        --dist-url tcp://${HOST}:${PORT} \
        --world_size ${NUM_GPU} \
        --rank 0 \
        --launcher pytorch \
        --results_dir ${RESULTS_DIR}

    # dataset_name: facial_components facial_attributes
    # test_type: fixed adaptive