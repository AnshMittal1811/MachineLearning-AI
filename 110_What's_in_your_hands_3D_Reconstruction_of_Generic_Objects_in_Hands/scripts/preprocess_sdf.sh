set -x 

DATA=$1
DATA_DIR=~/hoi/data
SDF_DIR=${DATA_DIR}/sdf/
INP_DIR=${DATA_DIR}/${DATA}
CODE_DIR=/private/home/yufeiy2/hoi/DeepSDF/


python scripts/preprocess_sdf.py ${DATA} ${DATA_DIR} ${SDF_DIR}

cd $CODE_DIR


# export PANGOLIN_WINDOW_URI=headless://

# python -m preprocess_data \
#     -s ${SDF_DIR}/MeshInp/${DATA} \
#     --split ${SDF_DIR}/MeshInp/${DATA}_all.json \
#     -d $SDF_DIR  \
#     --threads 16 --skip


# sanity check
# python -m datasets.sdf_img --config experiments/${DATA}.yaml