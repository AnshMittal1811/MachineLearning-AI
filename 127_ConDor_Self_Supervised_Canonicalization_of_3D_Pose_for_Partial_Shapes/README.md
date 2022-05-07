# ConDor: Self-Supervised Canonicalization of 3D Pose for Partial Shapes

*[Rahul Sajnani](https://arxiv.org/search/cs?searchtype=author&query=Sajnani%2C+R), [Adrien Poulenard](https://arxiv.org/search/cs?searchtype=author&query=Poulenard%2C+A), [Jivitesh Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+J), [Radhika Dua](https://arxiv.org/search/cs?searchtype=author&query=Dua%2C+R), [Leonidas J. Guibas](https://arxiv.org/search/cs?searchtype=author&query=Guibas%2C+L+J), [Srinath Sridhar](https://arxiv.org/search/cs?searchtype=author&query=Sridhar%2C+S)*

This is the official **TensorFlow and PyTorch** implementation for training and testing canonicalization results presented in ConDor.

![Teaser image](./images/teaser.jpg)



## Abstract

Progress in 3D object understanding has relied on manually canonicalized shape datasets that contain instances with consistent position and orientation (3D pose). This has made it hard to generalize these methods to in-the-wild shapes, eg., from internet model collections or depth sensors. ConDor is a self-supervised method that learns to Canonicalize the 3D orientation and position for full and partial 3D point clouds. We build on top of Tensor Field Networks (TFNs), a class of permutation- and rotation-equivariant, and translation-invariant 3D networks. During inference, our method takes an unseen full or partial 3D point cloud at an arbitrary pose and outputs an equivariant canonical pose. During training, this network uses self-supervision losses to learn the canonical pose from an un-canonicalized collection of full and partial 3D point clouds. ConDor can also learn to consistently co-segment object parts without any supervision. Extensive quantitative results on four new metrics show that our approach outperforms existing methods while enabling new applications such as operation on depth images and annotation transfer.     

## Dataset

Download the AtlasNetH5 dataset [here](https://condor-datasets.s3.us-east-2.amazonaws.com/dataset/ShapeNetAtlasNetH5_1024.zip).

```bash
# Create dataset directory
mkdir dataset
# Change directory
cd dataset
# Download the dataset (AtlasNet)
wget https://condor-datasets.s3.us-east-2.amazonaws.com/dataset/ShapeNetAtlasNetH5_1024.zip 
# Unzip the dataset
unzip ShapeNetAtlasNetH5_1024.zip 
```



## Pretrained models

Please find the pretrained are uploaded to the following links (For TensorFlow version only):

| Model        | Pretrained weights                                           |
| ------------ | ------------------------------------------------------------ |
| ConDor (F+P) | [ConDor (F+P) weights](https://drive.google.com/drive/folders/1nVLLeP1fv9JDN6U0oOLEoRSyMoVlJ4FH?usp=sharing) |
| ConDor (F)   | [ConDor (F) weights](https://drive.google.com/drive/folders/1pFTcwrsCM1iUSmfo8ppzf-0Vs-O7DVZD?usp=sharing) |



## TensorFlow implementation

Please find the TensorFlow implementation of ConDor and its instructions in the [ConDor](./ConDor) folder.



## PyTorch implementation

Please find the PyTorch implementation of ConDor and its instructions in the [ConDor_pytorch](./ConDor_pytorch) folder. This implementation is still under development.



## Citation

If you find this work helpful, please cite this

```
@InProceedings{sajnani2022_condor,
author = {Rahul Sajnani and
               Adrien Poulenard and
               Jivitesh Jain and
               Radhika Dua and
               Leonidas J. Guibas and
               Srinath Sridhar},
title = {ConDor: Self-Supervised Canonicalization of 3D Pose for Partial Shapes},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2022}
}

```





