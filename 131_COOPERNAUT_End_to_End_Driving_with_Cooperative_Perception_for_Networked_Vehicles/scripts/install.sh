#!/usr/bin/env bash

CONDA_NAME=autocast;

# Install dependencies
conda create -n ${CONDA_NAME} python=3.7 -y;
conda activate ${CONDA_NAME};
#conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y;
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install paho-mqtt scipy pygame py-trees==0.8.3 networkx==2.2 xmlschema numpy shapely imageio ray tqdm numba pandas scikit-image scikit-learn opencv-python h5py matplotlib;
conda install numba;
conda install -c hargup/label/pypi mosquitto -y;
pip install open3d==0.13.0;
conda install openblas-devel -c anaconda -y;
#pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include"  --install-option="--blas=openblas";
pip install wandb tensorboard torchsummary;

pip install torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html;
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html;
pip install torch-geometric==1.7.2;

# Install CPU Minkowski
DIR=$(pwd);
cd ..;
git clone https://github.com/NVIDIA/MinkowskiEngine.git;
cd MinkowskiEngine;
python setup.py install --cpu_only;
cd ${DIR};

# Download CARLA
DIR=$(pwd);
cd ..;
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz;
mkdir carla_0.9.11;
tar -xvzf CARLA_0.9.11.tar.gz -C carla_0.9.11;
# easy_install carla_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg;

# Setup environment variables
cd ${DIR}:
conda env config vars set CARLA_ROOT=${DIR}/../carla_0.9.11;
conda env config vars set SCENARIO_RUNNER_ROOT=${DIR}/AutoCastSim/srunner;
conda env config vars set PYTHONPATH=${DIR}/../carla_0.9.11/PythonAPI:${DIR}/../carla_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:${DIR}/../carla_0.9.11/PythonAPI/carla:${DIR}/AutoCastSim;

conda deactivate;
conda activate ${CONDA_NAME};
