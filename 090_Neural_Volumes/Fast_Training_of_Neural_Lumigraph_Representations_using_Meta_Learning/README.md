# Fast Training of Neural Lumigraph Representations using Meta Learning
### [Project Page](http://www.computationalimaging.org/publications/metanlr/) | [Paper](https://arxiv.org/abs/2106.14942) | [Data](https://drive.google.com/drive/folders/1dne-NheNYMPVhT4jV-76OiBLUXmLt1xm?usp=sharing)

[Alexander W. Bergman](http://alexanderbergman7.github.io),
[Petr Kellnhofer](https://kellnhofer.xyz/),
[Gordon Wetzstein](https://stanford.edu/~gordonwz/),
Stanford University. <br>
Official Implementation for Fast Training of Neural Lumigraph Representations using Meta Learning.

## Usage
To get started, create a conda environment with all dependencies:
```
conda env create -f environment.yml
conda activate metanlrpp
```

### Code Structure
The code is organized as follows:
- **experiment_scripts**: directory containing scripts to for training and testing MetaNLR++ models.
  - *pretrain_features.py*: pre-train encoder and decoder networks
  - *train_sdf_ibr_meta.py*: train meta-learned initialization for encoder, decoder, aggregation fn, and neural SDF
  - *test_sdf_ibr_meta.py*: specialize meta-learned initialization to a specific scene
  - *train_sdf_ibr.py*: train NLR++ model from scratch without meta-learned initialization
  - *test_sdf_ibr.py*: evaluate performance on withheld views
- **configs**: directory containing configs to reproduce experiments in the paper
  - *nlrpp_nlr.txt*: configuration for training NLR++ on the NLR dataset
  - *nlrpp_dtu.txt*: configuration for training NLR++ on the DTU dataset
  - *nlrpp_nlr_meta.txt*: configuration for training the MetaNLR++ initialization on the NLR dataset
  - *nlrpp_dtu_meta.txt*: configuration for training the MetaNLR++ initialization on the DTU dataset
  - *nlrpp_nlr_metaspec.txt*: configuration for training MetaNLR++ on the NLR dataset using the learned initialization
  - *nlrpp_dtu_metaspec.txt*: configuration for training MetaNLR++ on the DTU dataset using the learned initialization
- **data_processing**: directory containing utility functions for processing data
- **torchmeta**: torchmeta library for meta-learning
- **utils**: directory containing various utility functions for rendering and visualization
- *loss_functions.py*: file containing loss functions for evaluation
- *meta_modules.py*: contains meta learning wrappers around standard modules using torchmeta
- *modules.py*: contains standard modules for coodinate-based networks
- *modules_sdf.py*: extends standard modules for coordinate-based network representations of signed-distance functions.
- *modules_unet.py*: contains encoder and decoder modules used for image-space feature processing
- *scheduler.py*: utilities for training schedule
- *training.py*: training script
- *sdf_rendering.py*: functions for rendering SDF
- *sdf_meshing.py*: functions for meshing SDF
- **checkpoints**: contains checkpoints to some pre-trained models (additional/ablation models by request)
- **assets**: contains paths to checkpoints which are used as assets, and pre-computed buffers over multiple runs (if necessary)

### Getting Started

#### Pre-training Encoder and Decoder
Pre-train the encoder and decoder using the [FlyingChairsV2](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs2) training dataset as follows:
```
python experiment_scripts/pretrain_features.py --experiment_name XXX --batch_size X --dataset_path /path/to/FlyingChairs2/train
```
Alternatively, use the checkpoint in the checkpoints directory.

#### Training NLR++
Train a NLR++ model using the following command:
```
python experiment_scripts/train_sdf_ibr.py --config_filepath configs/nlrpp_dtu.txt --experiment_name XXX --dataset_path /path/to/dtu/scanXXX --checkpoint_img_encoder /path/to/pretrained/encdec
```

Note that we have uploaded our processed version of the DTU and NLR data [here](https://drive.google.com/drive/folders/1dne-NheNYMPVhT4jV-76OiBLUXmLt1xm?usp=sharing). The raw NLR data can be found [here](http://www.computationalimaging.org/publications/nlr/).

#### Meta-learned Initialization (MetaNLR++)
Meta-learn the initialization for the encoder, decoder, aggregation function, and neural SDF using the following command:
```
python experiment_scripts/train_sdf_ibr_meta.py --config_filepath configs/nlrpp_dtu_meta.txt --experiment_name XXX --dataset_path /path/to/dtu/meta/training --reference_view 24 --checkpoint_img_encoder /path/to/pretrained/encdec
```

Some optimized initializations for the DTU and NLR datasets can be found in the data directory. Additional models can be provided upon request.

#### Training MetaNLR++ from Initialization
Use the meta-learned initialization to specialize to a specific scene using the following command:
```
python experiment_scripts/test_sdf_ibr_meta.py --config_filepath configs/nlrpp_dtu_metaspec.txt --experiment_name XXX --dataset_path /path/to/dtu/scanXXX --reference_view 24 --meta_initialization /path/to/learned/meta/initialization
```

#### Evaluation
Test the converged scene on withheld views using the following command:
```
python experiment_scripts/test_sdf_ibr.py --config_filepath configs/nlrpp_dtu.txt --experiment_name XXX --dataset_path /path/to/dtu/scanXXX --checkpoint_path_test /path/to/checkpoint/to/evaluate
```

## Citation \& Contact
If you find our work useful in your research, please cite
```
@inproceedings{bergman2021metanlr,
author = {Bergman, Alexander W. and Kellnhofer, Petr and Wetzstein, Gordon},
title = {Fast Training of Neural Lumigraph Representations using Meta Learning},
booktitle = {NeurIPS},
year = {2021},
}
```

If you have any questions or would like access to specific ablations or baselines presented in the paper or supplement (the code presented here is only a subset based off of the source code used to generate the results), please feel free to contact the authors. Alex can be contacted via e-mail at awb@stanford.edu. 
