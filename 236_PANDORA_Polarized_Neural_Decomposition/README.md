# PANDORA: Polarization-Aided Neural Decomposition Of Radiance
### [Project Page](https://akshatdave.github.io/pandora) | [Paper](https://arxiv.org/abs/2203.13458) | [Data](https://drive.google.com/file/d/1FvOi_2wfSUnASHulQdBhHQcQCxOuJ8zz/view?usp=sharing)
<br>
Official PyTorch implementation of Pandora: an inverse rendering technique that exploits neural implicit representations and polarization cues. 

<br>

[PANDORA: Polarization-Aided Neural Decomposition Of Radiance](https://akshatdave.github.io/pandora)

 [Akshat Dave](https://akshadave.github.io),
 [Yongyi Zhao](https://yongyizhao.com/),
 [Ashok Veeraraghavan](https://computationalimaging.rice.edu/team/ashok-veeraraghavan/) 

 [Computational Imaging Lab, Rice University](https://computationalimaging.rice.edu)

accepted for ECCV 2022

![Teaser Animation](media/teaser_animation.gif)

## Setting up
### Loading conda environment

Create a new Anaconda environment using the supplied `environment.yml` 
```
conda env create -f environment.yml
```

### Downloading datasets

Unzip [this](https://drive.google.com/file/d/1FvOi_2wfSUnASHulQdBhHQcQCxOuJ8zz/view?usp=sharing) zip file (4.5 GB) into the `data` folder of the repo directory. The zip file contains real and rendered multi-view polarimetric datasets shown in the paper. 

Refer to `dataio/Ours.py` and `data/Mitsuba2.py` for pre-processing of real and rendered data respectively.

## Training

Run the following command to train geometry and radiance neural representations from multi-view polarimetric images.
```
python -m train --config configs/real_ceramic_owl.yaml
```
Config files input through `--config` describe the parameters required for training. As an example the parameters for real ceramic owl dataset are described in `real_ceramic_owl.yaml`

Tensorboard logs, checkpoints, arguments and images are saved in the corresponding experiment folder in `logs/`.

## Rendering Trained Representations
Using the saved arguments from `config.yaml` and the saved checkpoint such as `latest.pt` in `logs/`, novel views can be rendered using the following command.
```
python -m tools.render_view 
    --config  logs/our_ceramic_owl_v2/config.yaml
    --load_pt logs/our_ceramic_owl_v2/ckpts/latest.pt
```
Refer to [this](https://github.com/ventusff/neurecon/blob/main/docs/usage.md#evaluation-free-viewport-rendering) documentation in  `neurecon` repo for possible camera trajectories. By default first three views used for training are rendered. 

Outputs are saved in the corresponding experiment folder in `out/`. By default,the outputs include surface normal, diffuse radiance, specular radiance and combined radiance for each view along with the estimated roughness.

## Acknowledgements

This repository adapts code or draws inspiration from

- https://github.com/ventusff/neurecon
- https://github.com/yenchenlin/nerf-pytorch
- https://github.com/Fyusion/LLFF
- https://github.com/elerac/polanalyser
- https://github.com/sxyu/svox2

## Citation

```
@article{dave2022pandora,
  title={PANDORA: Polarization-Aided Neural Decomposition Of Radiance},
  author={Dave, Akshat and Zhao, Yongyi and Veeraraghavan, Ashok},
  journal={arXiv preprint arXiv:2203.13458},
  year={2022}
}
```