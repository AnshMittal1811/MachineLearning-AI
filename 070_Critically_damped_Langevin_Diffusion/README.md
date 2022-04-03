<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/score-based-generative-modeling-with-1/image-generation-on-cifar-10)](https://paperswithcode.com/sota/image-generation-on-cifar-10?p=score-based-generative-modeling-with-1)

# <p align="center">Score-Based Generative Modeling <br> with Critically-Damped Langevin Diffusion <br><br> ICLR 2022 (spotlight)</p>

<div align="center">
  <a href="https://timudk.github.io/" target="_blank">Tim&nbsp;Dockhorn</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://latentspace.cc/" target="_blank">Arash&nbsp;Vahdat</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://karstenkreis.github.io/" target="_blank">Karsten&nbsp;Kreis</a>
  <br> <br>
  <a href="https://arxiv.org/abs/2112.07068" target="_blank">Paper</a> &emsp;
  <a href="https://nv-tlabs.github.io/CLD-SGM/" target="_blank">Project&nbsp;Page</a> 
</div>
<br><br>
<p align="center">
    <img width="750" alt="Animation" src="assets/animation.gif"/>
</p>

## Requirements

CLD-SGM is built in Python 3.8.0 using PyTorch 1.8.1 and CUDA 11.1. Please use the following command to install the requirements:
```shell script
pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://storage.googleapis.com/jax-releases/jax_releases.html
``` 
Optionally, you may also install [NVIDIA Apex](https://github.com/NVIDIA/apex). The Adam optimizer from this library is faster than PyTorch's native Adam.

## Preparations

CIFAR-10 does not require any data preparation as the data will be downloaded directly. To download CelebA-HQ-256 and prepare the dataset for training models, please run the following lines:

```shell script
mkdir -p data/celeba/
wget -P data/celeba/ https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar
tar -xvf data/celeba/celeba-tfr.tar -C data/celeba/
python util/convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=data/celeba/celeba-tfr --lmdb_path=data/celeba/celeba-lmdb --split=train
python util/convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=data/celeba/celeba-tfr --lmdb_path=data/celeba/celeba-lmdb --split=validation
```

For multi-node training, the following environment variables need to be specified: `$IP_ADDR` is the IP address of the machine that will host the process with rank 0 during training (see [here](https://pytorch.org/tutorials/intermediate/dist_tuto.html#initialization-methods)). `$NODE_RANK` is the index of each node among all the nodes.

## Checkpoints

We provide pre-trained CLD-SGM checkpoints for CIFAR-10 and CelebA-HQ-256 [here](https://drive.google.com/drive/folders/1tgYRqCWAq1YDKe7zh5nkffpFA55yjqBr?usp=sharing).

## Training and evaluation

<details><summary>CIFAR-10</summary>

- Training our CIFAR-10 model on a single node with one GPU and batch size 64:

```shell script
python main.py -cc configs/default_cifar10.txt -sc configs/specific_cifar10.txt --root $ROOT --mode train --workdir work_dir/cifar10 --n_gpus_per_node 1 --training_batch_size 64 --testing_batch_size 64 --sampling_batch_size 64
```

Hidden flags can be found in the config files: `configs/default_cifar10.txt` and `configs/specific_cifar10.txt`. The flag `--sampling_batch_size` indicates the batch size per GPU, whereas `--training_batch_size` and `--eval_batch_size` indicate the total batch size of all GPUs combined. The script will update a running checkpoint every `--snapshot_freq` iterations (saved, in this case, at `work_dir/cifar10/checkpoints/checkpoint.pth`), starting from `--snapshot_threshold`. In `configs/specific_cifar10.txt`, these values are set to 10000 and 1, respectively.

- Training our CIFAR-10 model on two nodes with 8 GPUs each and batch size 128:
```shell script
mpirun --allow-run-as-root -np 2 -npernode 1 bash -c 'python main.py -cc configs/default_cifar10.txt -sc configs/specific_cifar10.txt --root $ROOT --mode train --workdir work_dir/cifar10 --n_gpus_per_node 8 --training_batch_size 8 --testing_batch_size 8 --sampling_batch_size 128 --node_rank $NODE_RANK --n_nodes 2 --master_address $IP_ADDR'
```

- To resume training, we simply change the mode from train to continue (two nodes of 8 GPUs):
```shell script
mpirun --allow-run-as-root -np 2 -npernode 1 bash -c 'python main.py -cc configs/default_cifar10.txt -sc configs/specific_cifar10.txt --root $ROOT --mode continue --workdir work_dir/cifar10 --n_gpus_per_node 8 --training_batch_size 8 --testing_batch_size 8 --sampling_batch_size 128 --cont_nbr 1 --node_rank $NODE_RANK --n_nodes 2 --master_address $IP_ADDR'
```

Any file within `work_dir/cifar10/checkpoints/` can be used to resume training by setting `--checkpoint` to the particular file name. If `--checkpoint` is unspecified, the script automatically uses the last snapshot checkpoint (`checkpoint.pth`) to continue training. The flag `--cont_nbr` makes sure that a new random seed is used for training continuation; for additional continuation runs `--cont_nbr` may be incremented by one.

- The following command can be used to evaluate the negative ELBO as well as the FID score (two nodes of 8 GPUs):
```shell script
mpirun --allow-run-as-root -np 2 -npernode 1 bash -c 'python main.py -cc configs/default_cifar10.txt -sc configs/specific_cifar10.txt --root $ROOT --mode eval --workdir work_dir/cifar10 --n_gpus_per_node 8 --training_batch_size 8 --testing_batch_size 8 --sampling_batch_size 128 --eval_folder eval_elbo_and_fid --ckpt_file checkpoint_file --eval_likelihood --eval_fid --node_rank $NODE_RANK --n_nodes 2 --master_address $IP_ADDR'
```

Before running this you need to download the FID stats file from [here](https://drive.google.com/file/d/14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI/view?usp=sharing) and place it into `$ROOT/assets/stats/`).

To evaluate our provided CIFAR-10 model download the checkpoint [here](https://drive.google.com/drive/folders/1KcELOxCOATj3zr_aVfdFIFzXw8Bzev4h?usp=sharing), create a directory `work_dir/cifar10_pretrained_seed_0/checkpoints`, place the checkpoint in it, and set `--ckpt_file checkpoint_800000.pth` as well as `--workdir cifar10_pretrained`.

</details>

<details><summary>CelebA-HQ-256</summary>

- Training the CelebA-HQ-256 model from our paper (two nodes of 8 GPUs and batch size 64):
```shell script
mpirun --allow-run-as-root -np 2 -npernode 1 bash -c 'python main.py -cc configs/default_celeba_paper.txt -sc configs/specific_celeba_paper.txt --root $ROOT --mode train --workdir work_dir/celeba_paper --n_gpus_per_node 8 --training_batch_size 4 --testing_batch_size 4 --sampling_batch_size 64 --data_location data/celeba/celeba-lmdb/ --node_rank $NODE_RANK --n_nodes 2 --master_address $IP_ADDR'
```

We found that training of the above model can potentially be unstable. Some modifications that we found post-publication lead to better numerical stability as well as improved performance:
```shell script
mpirun --allow-run-as-root -np 2 -npernode 1 bash -c 'python main.py -cc configs/default_celeba_post_paper.txt -sc configs/specific_celeba_post_paper.txt --root $ROOT --mode train --workdir work_dir/celeba_post_paper --n_gpus_per_node 8 --training_batch_size 4 --testing_batch_size 4 --sampling_batch_size 64 --data_location data/celeba/celeba-lmdb/ --node_rank $NODE_RANK --n_nodes 2 --master_address $IP_ADDR'
```

In contrast to the model reported in our paper, we make use of a non-constant time reparameterization function &beta;(t). For more details, please check the config files.

</details>

<details><summary>Toy data</summary>

- Training on the multimodal Swiss Roll dataset using a single node with one GPU and batch size 512:

```shell script
python main.py -cc configs/default_toy_data.txt --root $ROOT --mode train --workdir work_dir/multi_swiss_roll --n_gpus_per_node 1 --training_batch_size 512 --testing_batch_size 512 --sampling_batch_size 512 --dataset multimodal_swissroll
```

Additional toy datasets can be implemented in `util/toy_data.py`.

</details>

## Monitoring the training process

We use Tensorboard to monitor the progress of training. For example, monitoring the CIFAR-10 process can be done as follows:
```shell script
tensorboard --logdir work_dir/cifar10_seed_0/tensorboard
```

## Demonstration

Load our pretrained checkpoints and play with sampling and likelihood computation:

| Link | Description|
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nixYLeGQZd5-sY-s_pSvTCtPcEUlaj05?usp=sharing)  | CIFAR-10 |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d84GfWd_6Fu52Ujtg8shGYUgbCfBDies?usp=sharing) | CelebA-HQ-256 |

## Citation
If you find the code useful for your research, please consider citing our ICLR paper:

```bib
@inproceedings{dockhorn2022score,
  title={Score-Based Generative Modeling with Critically-Damped Langevin Diffusion},
  author={Tim Dockhorn and Arash Vahdat and Karsten Kreis},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```

## License

Copyright © 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License. Please see our main [LICENSE](./LICENSE) file.

#### License Dependencies

For any code dependencies related to StyleGAN2, the license is the  Nvidia Source Code License-NC by NVIDIA Corporation, see [StyleGAN2 LICENSE](https://nvlabs.github.io/stylegan2/license.html).

This code it built on the excellent ScoreSDE codebase by Song et al., which can be found [here](https://github.com/yang-song/score_sde_pytorch). For any code dependencies related to ScoreSDE, the license is the Apache License 2.0, see [ScoreSDE LICENSE](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE). -->
