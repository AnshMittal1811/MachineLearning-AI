# Modeling Indirect Illumination for Inverse Rendering

### [Project Page](https://zju3dv.github.io/invrender) | [Paper](https://arxiv.org/pdf/2204.06837.pdf) | [Data](https://drive.google.com/file/d/1wWWu7EaOxtVq8QNalgs6kDqsiAm7xsRh/view?usp=sharing)



## Preparation
- Set up the python environment

```sh
conda create -n invrender python=3.7
conda activate invrender

pip install -r requirement.txt
```

- Download our example synthetic dataset from [Google Drive](https://drive.google.com/file/d/1wWWu7EaOxtVq8QNalgs6kDqsiAm7xsRh/view?usp=sharing)


## Run the code

#### Training

Taking the scene `hotdog` as an example, the training process is as follows.

1. Optimize geometry and outgoing radiance field from multi-view images. (Same as [IDR](https://github.com/lioryariv/idr))

   ```sh
   cd code
   python training/exp_runner.py --conf confs_sg/default.conf \
                                 --data_split_dir ../Synthetic4Relight/hotdog \
                                 --expname hotdog \
                                 --trainstage IDR \
                                 --gpu 1
   ```

2. Draw sample rays above surface points to train the indirect illumination and visibility MLP.

   ```sh
   python training/exp_runner.py --conf confs_sg/default.conf \
                                 --data_split_dir ../Synthetic4Relight/hotdog \
                                 --expname hotdog \
                                 --trainstage Illum \
                                 --gpu 1
   ```
   
3. Jointly optimize diffuse albedo, roughness and direct illumination.

   ```sh
   python training/exp_runner.py --conf confs_sg/default.conf \
                                 --data_split_dir ../Synthetic4Relight/hotdog \
                                 --expname hotdog \
                                 --trainstage Material \
                                 --gpu 1
   ```

#### Relighting

- Generate videos under novel illumination.

  ```sh
  python scripts/relight.py --conf confs_sg/default.conf \
                            --data_split_dir ../Synthetic4Relight/hotdog \
                            --expname hotdog \
                            --timestamp latest \
                            --gpu 1
  ```

## Citation

```
@inproceedings{zhang2022invrender,
  title={Modeling Indirect Illumination for Inverse Rendering},
  author={Zhang, Yuanqing and Sun, Jiaming and He, Xingyi and Fu, Huan and Jia, Rongfei and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2022}
}
```

Acknowledgements: part of our code is inherited from  [IDR](https://github.com/lioryariv/idr) and [PhySG](https://github.com/Kai-46/PhySG). We are grateful to the authors for releasing their code.

