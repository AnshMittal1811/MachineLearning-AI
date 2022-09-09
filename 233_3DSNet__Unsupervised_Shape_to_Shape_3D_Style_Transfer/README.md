# 3DSNet: Unsupervised Shape-to-shape 3D Style Transfer
This repository contains the code for our method for learning-based 3D style transfer as described in ["3DSNet: Unsupervised Shape-to-shape 3D Style Transfer"](https://arxiv.org/abs/2011.13388). The code used to train and evaluate our framework on the Shapenet dataset is here provided and ready to use.

If you find this code useful for your project, please consider citing our paper:

```bibtex
@article{segu20203dsnet,
  title={3DSNet: Unsupervised Shape-to-Shape 3D Style Transfer},
  author={Segu, Mattia and Grinvald, Margarita and Siegwart, Roland and Tombari, Federico},
  journal={arXiv preprint arXiv:2011.13388},
  year={2020}
}
```

![Reconstruction and style transfer results with 3DSNet on the archair-chair category.](docs/chairs.jpg)

## Video
A video of our results will soon be available.

## Prerequisites
### Install
This implementation uses Python 3.6, [Pytorch](http://pytorch.org/), [Pymesh](https://github.com/PyMesh/PyMesh), Cuda 10.1. 

```shell
# Copy-Paste the snippet in a terminal
git clone --recurse-submodules https://github.com/ethz-asl/3dsnet.git
cd 3dsnet 

# Install dependencies
conda create -n 3dsnet python=3.6 --yes
conda activate 3dsnet

conda install  pytorch torchvision cudatoolkit=10.1 -c pytorch --yes
conda install -y -c conda-forge pyembree
conda install -y -c conda-forge trimesh seaborn
conda install -c conda-forge -c fvcore fvcore
conda install pytorch3d -c pytorch3d

pip install git+https://github.com/rtqichen/torchdiffeq torchvision
pip3 install git+https://github.com/cnr-isti-vclab/PyMeshLab
pip install --user --requirement  requirements.txt # pip dependencies

```

Chumpy installation with pip is currently broken with pip version 20.1. Please use pip 20.0.2 until chumpy issue won't be fixed.

### Compile Chamfer (MIT) + Metro Distance (GPL3 Licence)
```shell
# Copy/Paste the snippet in a terminal
python auxiliary/ChamferDistancePytorch/chamfer3D/setup.py install #MIT
cd auxiliary
git clone https://github.com/ThibaultGROUEIX/metro_sources.git
cd metro_sources; python setup.py --build # build metro distance #GPL3
cd ../..

```

## Before running the code
### Auxiliary models
Please download all auxiliary models needed for our framework in the `aux_models` folder [HERE](https://drive.google.com/drive/folders/1cyVRUmtN_YF-TXkytKfn1M0HlGH9Qux_?usp=sharing) and place them at `.../3dsnet/aux_models`.

### Data
The pre-trained model publicly provided are trained on the [ShapeNet dataset](https://www.shapenet.org/). The pointcloud version of the dataset is automatically downloaded when running the code for the first time. 

To seamlessly run the code, please also download the ShapeNet Core V1 dataset from [HERE](https://www.shapenet.org/) and move it to the subdirectory `.../ShapeNet` of the chosen `opt.data_dir`, set as default in the [argument_parser.py](auxiliary/argument_parser.py) to `dataset/data/`. 
In the same folder, also put the `all.csv` file containing training, validation and test splits for ShapeNet Core V1. You can download it from [HERE](https://drive.google.com/drive/folders/18OxvcDcCoxAfypU0zDrhsnCA4yXG_NRL?usp=sharing) or from the [original page](https://www.shapenet.org/) if available in the download section. Please notice that, despite V2 being already available, we used ShapeNet V1 for compatibility with the pointcloud version originally provided with the [official Atlasnet implementation](https://github.com/ThibaultGROUEIX/AtlasNet).

## Pre-trained Models
You can find pre-trained models for our framework in the `3dsnet_models` folder [HERE](https://drive.google.com/drive/folders/1cyVRUmtN_YF-TXkytKfn1M0HlGH9Qux_?usp=sharing).

## Running the code
You can play with different parameters configurations by changing them directly in the provided training/evaluation/demo scripts or in the [argument_parser.py](auxiliary/argument_parser.py).

For further details, please refer to the parameters description in the [argument_parser.py](auxiliary/argument_parser.py).

### Training
You can easily start training 3DSNet launching the provided training scripts:
```
./train_chairs.sh
```
or

```
./train_planes.sh
```

### Evaluation
You can easily start evaluation of a pretrained model launching the provided training scripts:
```
./eval.sh
```

Please, modify the parameter `RELOAD_MODEL_PATH` according to the model that you wish to evaluate.

### Demo
The provided demo script allows to generate multiple 3D objects and their interpolation in the latent space from a pretrained model.
```
./demo.sh
```

Please, modify the parameter `RELOAD_MODEL_PATH` according to the model that you wish to evaluate.

### Acknowledgments
The code in this repository is built on the [official implementation of Atlasnet](https://github.com/ThibaultGROUEIX/AtlasNet).

The implementation of our adaptive Meshflow decoder is based on the [official Meshflow implementation](https://github.com/KunalMGupta/NeuralMeshFlow).


If you cite our work, please consider citing also theirs.
```bibtex
@inproceedings{groueix2018papier,
  title={A papier-m{\^a}ch{\'e} approach to learning 3d surface generation},
  author={Groueix, Thibault and Fisher, Matthew and Kim, Vladimir G and Russell, Bryan C and Aubry, Mathieu},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={216--224},
  year={2018}
}
```

```bibtex
@article{gupta2020neural,
  title={Neural mesh flow: 3d manifold mesh generationvia diffeomorphic flows},
  author={Gupta, Kunal and Chandraker, Manmohan},
  journal={arXiv preprint arXiv:2007.10973},
  year={2020}
}
```
