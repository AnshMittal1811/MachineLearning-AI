
# Download utility for SLIM dataset
# bash bin/download.sh <dataset-name>

# Dataset name mast be one of the following
# * turk_data
# * synthetic_data

# Kwargs
export DATASET_NAME=${1:-turk_data}

# Path
export DATA_DIR=./data/
export DATASET_DIR=${DATA_DIR}/${DATASET_NAME}/

# Check gsutil command
if type gsutil > /dev/null 2>&1; then
    echo "gsutil command does exist"
else
    echo "gsutil command does not exist"
    exit 1
fi

# Check data dir
if [[ ! -d ${DATA_DIR} ]]; then
    echo "Make data dir"
    mkdir ${DATA_DIR}
fi

# Download dataset
if [[ ! -d ${DATASET_DIR} ]]; then
    echo "Download dataset"
    gsutil -m cp -r gs://slim-dataset/${DATASET_NAME}/ ./${DATA_DIR}/

    # Make train/valid/test dir for turk_data
    if [[ ${DATASET_NAME} = "turk_data" ]]; then
        echo "Make train/valid/test dir"
        cd ${DATASET_DIR}
        mkdir train valid test
        mv train.tfrecord train/train.tfrecord
        mv valid.tfrecord valid/valid.tfrecord
        mv test.tfrecord test/test.tfrecord
        cd ../..
    fi
else
    echo "Specified dataset already exists"
fi

echo "Convert tfrecord to gzip files"

# Convert tfrecords to gzip files
python3 ./examples/convert_slim_dataset.py --dataset ${DATASET_NAME} \
    --mode train --first-n -1 --batch-size 500

python3 ./examples/convert_slim_dataset.py --dataset ${DATASET_NAME} \
    --mode valid --first-n -1 --batch-size 500

python3 ./examples/convert_slim_dataset.py --dataset ${DATASET_NAME} \
    --mode test --first-n -1 --batch-size 500

echo "Completed"
