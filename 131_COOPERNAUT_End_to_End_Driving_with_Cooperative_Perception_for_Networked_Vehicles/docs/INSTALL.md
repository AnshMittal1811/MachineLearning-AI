## Installation

* First, clone the repo:

```bash
git clone --recursive git@github.com:UT-Austin-RPL/Coopernaut.git
```

Next, please follow the instructions below to setup Coopernaut.

The easiest way to do this is through [conda](https://docs.anaconda.com/anaconda/install/index.html):

```bash
conda env create -n autocast python=3.7
```

### Installing dependencies
Install the following under the `autocast` conda environment
```bash
conda activate autocast
```
* Install PyTorch: `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch`
* Install pip dependencies: `pip3 install paho-mqtt scipy pygame py-trees==0.8.3 networkx==2.2 xmlschema numpy shapely imageio ray tqdm numba pandas scikit-image scikit-learn opencv-python h5py`
* Install mosquitto: `conda install -c hargup/label/pypi mosquitto`
* Install torch-scatter and torch-sparse: 
```bash
pip install torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install torch-sparse==0.6.10 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install torch-geometric==1.7.2
```
* Install open3d: `pip install open3d[None]==0.13.0`
* Install MinkowskiEngine: 
```bash
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine@f81ae66b33b883cd08ee4f64d08cf633608b118 -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include"  --install-option="--blas=openblas"
```
* Install logging tools
```bash
pip install wandb tensorboard torchsummary
```
* Install setuptools
```bash
conda install setuptools
```

### Installing CARLA
* Download [CARLA 0.9.11](https://github.com/carla-simulator/carla/releases/tag/0.9.11)
* Install the python egg using `easy_install` inside `PythonAPI/carla/dist` of the CARLA repo.

### Specifying environment variables
Configure the following environment variables:

```bash
export CARLA_ROOT=[LINK TO YOUR CARLA FOLDER]
export SCENARIO_RUNNER_ROOT=[LINK TO AUTOCASTSIM]/srunner
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:[LINK TO AUTOCASTSIM]
```
