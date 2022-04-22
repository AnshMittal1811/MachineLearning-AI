# Installation


- [~5min] The basic installation, including pytorch, other dependency, etc. 
  We use pytorch 1.8.1, cuda version 10.2, pytorch-lightning 1.2 (there will be some incompatible issue about slurm preemption for version above 1.2)
  ```
  git clone https://github.com/JudyYe/ihoi.git
  cd ihoi
  conda env create -f docs/env.yaml
  conda activate ihoi
  ```


- [~3min] Install other third-party libraries including: [FrankMocap](https://github.com/facebookresearch/frankmocap/) with [my minor modification](https://github.com/judy/frankmocap/), [MANO](https://github.com/hassony2/manopth), [Detectron2](https://github.com/facebookresearch/detectron2) and   download our pretrained models. 
  - Run the following script
  ```
  sh scripts/one_click.sh
  ```
  - If any step doesn't work, please follow the instruction from the official website.

- Setting SMPL-X/MANO Models
    - We use use SMPL-X and MANO model as 3D pose estimation output. You have to accept the license and download them from the original website.
    - Download SMPLX Model (Neutral model: SMPLX_NEUTRAL.pkl):
        - You can use SMPL-X model for body mocap instead of SMPL model. 
        - Download ```SMPLX_NEUTRAL.pkl``` in the original [SMPL website](https://smpl-x.is.tue.mpg.de/). You need to register to download the SMPLX data.
        - Put the ```SMPLX_NEUTRAL.pkl``` file in: ./externals/frankmocap/extra_data/smpl/SMPLX_NEUTRAL.pkl
        - This is for hand module and whole body module
    - Download MANO Model (Neutral model: MANO_LEFT.pkl, MANO_RIGHT.pkl):
        - Download ```Models & Code``` in the original [MANO website](https://mano.is.tue.mpg.de/). You need to register to download the MANO data.
        - Put the ```models/MANO_LEFT.pkl``` ```models/MANO_RIGHT.pkl``` file in: `./externals/mano/`

    - Once you sucessfully installed and downloaded all, you should have the following files in your directory, here is a list of key files:
        ```
        ./externals
        |── detectron2
        |── mano
        |   |── MANO_LEFT.pkl    
        |   └── MANO_RIGHT.pkl
        └── frankmocap
            └── extra_data
                └── smpl
                    |── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
                    └── SMPLX_NEUTRAL.pkl

        ```

### Acknowledgement
Structure of this installation instruction is inspired from [FrankMocap](https://github.com/facebookresearch/frankmocap/blob/main/docs/INSTALL.md).