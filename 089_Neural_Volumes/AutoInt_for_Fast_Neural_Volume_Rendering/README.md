# AutoInt: Automatic Integration for Fast Neural Volume Rendering <br> CVPR 2021
### [Project Page](http://www.computationalimaging.org/publications/automatic-integration/) | [Video](https://www.youtube.com/watch?v=GYxFYbih0PU) | [Paper](https://arxiv.org/abs/2012.01714)
[![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/computational-imaging/automatic-integration/blob/master/autoint_example.ipynb)<br>
PyTorch implementation of automatic integration.<br>
[AutoInt: Automatic Integration for Fast Neural Volume Rendering](http://www.computationalimaging.org/publications/automatic-integration/)<br>
 [David B. Lindell](https://davidlindell.com)\*,
 [Julien N. P. Martel](http://web.stanford.edu/~jnmartel/)\*,
 [Gordon Wetzstein](https://computationalimaging.org)<br>
 Stanford University <br>
  \*denotes equal contribution  
in CVPR 2021

<img src='imgs/rendering.jpg'/>

## Quickstart

To get started quickly, we provide a colab link above. Otherwise, you can clone this repo and follow the below instructions. 

To setup a conda environment, download example training data, begin the training process, and launch Tensorboard:
```
conda env create -f environment.yml
conda activate autoint 
cd experiment_scripts
python train_1d_integral.py
tensorboard --logdir=../logs --port=6006
```

This example will fit a grad network to a 1D signal and evaluate the integral. You can monitor the training in your browser at `localhost:6006`. You can also train a network on the sparse tomography problem presented in the paper with `python train_sparse_tomography.py`.  


### Autoint for Neural Rendering

Automatic integration can be used to learn closed form solutions to the volume rendering equation, which is an integral equation accumulates transmittance and emittance along rays to render an image. While conventional neural renderers require hundreds of samples along each ray to evaluate these integrals (and hence hundreds of costly forward passes through a network), AutoInt allows evaluating these integrals far fewer forward passes. 

#### Training

To run AutoInt for neural rendering, first set up the conda environment with  
```
conda env create -f environment.yml
conda activate autoint 
```

Then, download the datasets to the `data` folder. We allow training on any of three datasets. The synthetic Blender data from [NeRF](https://github.com/bmild/nerf) and the [LLFF](https://github.com/Fyusion/LLFF) scenes are hosted [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). The [DeepVoxels](https://github.com/vsitzmann/deepvoxels) data are hosted [here](https://drive.google.com/open?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl).

Finally, use the provided config files in the `experiment_scripts/configs` folder to train on these datasets. For example, to train on a NeRF Blender dataset, run the following
```
python train_autoint_radiance_field.py --config ./configs/config_blender_tiny.ini
tensorboard --logdir=../logs/ --port=6006
```

This will train a small, low-resolution scene. To train scenes at high-resolution (requires a few days of training time), use the `config_blender.ini`, `config_deepvoxels.ini`, or `config_llff.ini` config files. 

#### Rendering

Rendering from a trained model can be done with the following command.
```
python train_autoint_radiance_field.py --config /path/to/config/file --render_model ../logs/path/to/log/directory <epoch number> --render_ouput /path/to/output/folder
```

Here, the `--render_model` command indicates the log directory where the code saves the models and checkpoints. For example, this would be `../logs/blender_lego` for the default Blender dataset. Then, the epoch number can be found by looking at numbers of the the saved checkpoint filenames in `../logs/blender_lego/checkpoints/`. Finally, `--render_output` should specify a folder where the output rendered images will be generated.

## Citation

```
@inproceedings{autoint2021,
  title={AutoInt: Automatic Integration for Fast Neural Volume Rendering},
  author={David B. Lindell and Julien N. P. Martel and Gordon Wetzstein},
  year={2021},
  booktitle={Proc. CVPR},
}
```
