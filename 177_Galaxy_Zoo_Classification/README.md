# Galaxy Morphological Classification with Efficient Vision Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2110.01024-yellowgreen.svg)](https://arxiv.org/abs/2110.01024)

Repository of the paper **Galaxy Morphological Classification with Efficient Vision Transformer** by Joshua Yao-Yu Lin, Song-Mao Liao, Hung-Jin Huang, Wei-Ting Kuo, and Olivia Hsuan-Min Ou.

## Abstract

Quantifying the morphology of galaxies has been an important task in astrophysics to understand the formation and evolution of galaxies. In recent years, the data size has been dramatically increasing due to several on-going and upcoming surveys.

In this work, we explore the usage of [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT) for galaxy morphology classification for the first time.

We show that ViT could reach competitive results compared with CNNs, and is specifically good at classifying smaller-sized and fainter galaxies.

[![vit-oo.png](https://i.postimg.cc/dt1gky53/vit-oo.png)](https://postimg.cc/F1MTw75X )
<p align="center">
    The architecture overview of Vision Transformer. This diagram is adapted from
    https://arxiv.org/abs/2010.11929
</p>

## Results

We use [Linformer](https://arxiv.org/abs/2006.04768) as our Vision Transformer model. The main feature of Linformer is its linear (O(n), where n is the size of the input patch sequence) attention complexity instead of the quadratic complexity (O(n<sup>2</sup>)) in the original ViT. This reduction of complexity is essential particularly for lowering computational cost. This efficient training originates from approximating the original attention matrix by a low-rank matrix.

[![attention-formula.png](https://i.postimg.cc/9QX6WM3Z/attention-formula.png)](https://postimg.cc/pp7GCPLL)
<p align="center">
    The calculation of the attention in Linformer
</p>

We present our best overall accuracy and individual class accuracy from our Linformer models. Due to
the intrinsic imbalance in different categories, categorical accuracy is another important performance
indicator.

Our best overall accuracy is 80:55%, whereas the best individual class accuracy achieved in our weighted-cross entropy Linformer is over 60% in each class (the overall accuracy is 77:42%).

All their individual class accuracy results are shown in the confusion matrix (below).

[![vit-fig3.png](https://i.postimg.cc/cLkG81N7/vit-fig3.png)](https://postimg.cc/cv3bpSC6)
<p align="center">
    Confusion matrices of the ViT model, with uniform (left) and tuned (right) class weights
</p>

We find that ViT performs better than CNN in classifying smaller and fainter galaxies correctly which are more challenging to classify since they are noisier. A possible reasoning for ViT’s better performance on fainter and smaller galaxies is that these galaxies dominate the entire dataset and ViT models tend to outperform CNN when more training samples are available.

[![vit-fig4.png](https://i.postimg.cc/d0fSmnnD/vit-fig4.png)](https://postimg.cc/qt8LpsPd)
<p align="center">
    ViT is specifically better at classifying smaller-sized and fainter galaxies
</p>

Figures used in the paper can be produced by the notebooks in the directory `figures/`.

## Dataset

[The Galaxy Zoo Dataset](https://data.galaxyzoo.org/)

The galaxy dataset used in this study is based on the Galaxy Zoo 2 Project2 (GZ2), with the morphological information drawn from the catalog of [Hart et al.](https://arxiv.org/abs/1607.01019), and the galaxy images downloaded from [kaggle](https://www.kaggle.com/jaimetrickz/galaxy-zoo-2-images).

The size of each image is `shape = (424, 424, 3)`, with the color channels corresponding the g, r, i filters of the SDSS.

We construct a clean galaxy dataset with eight distinct classes and label them from 0 to 7 in the order of:
round elliptical, in-between elliptical, cigar-shaped elliptical, edge-on, barred spiral, unbarred spiral,
irregular and merger galaxies:

[![vit-fig2.png](https://i.postimg.cc/Ls5zjjJC/vit-fig2.png)](https://postimg.cc/HJGcgc4X)
<p align="center">
    Examples of galaxy images from each of the eight morphological classes
</p>

Our final baseline dataset consists of 155,951 images. We split the data into 64% train set, 16% validation set, and 20% test set. We crop images into `shape = (224, 224, 3)` and use data augmentation techniques by flipping and rotating the images.

Original images of all galaxies can be found in the [Galaxy Zoo Dataset](https://data.galaxyzoo.org/). The galaxy IDs and their corresponding class labels are stored in `gz2_data/gz2_train.csv`, `gz2_data/gz2_valid.csv`, and `gz2_data/gz2_test.csv`.

## Code

The code for training the ViT model can be found in the Jupyter Notebook: `gz2_ViT_Linformer_Pytorch.ipynb`

The code for training CNN models can be found in `gz2_Resnet50_Pytorch.ipynb` or `gz2_VGG16bn_Pytorch.ipynb`

The directory `data_preprocess` contains the codes used for data preprocessing and analysis of class composition.

`gz2_pretrained_ViT.ipynb` uses pretrained ViT (from HuggingFace) and achieves a better performance.

`gz2_ViT_attention_maps.ipynb` generates the attentions maps of the galaxy images


## Hyperparameters

Below lists the hyperparameters used in training.

**ViT Linformer**
```
PATCH_SIZE = 28
DEPTH = 12
HIDDEN_DIM = 128
K_DIM = 64
NUM_HEADS = 8

LR = 3e-4
STEP_SIZE = 5
GAMMA = 0.9
MAX_EPOCH = 200

LIN_DROPOUT = 0.1

class_weights = [1., 1., 1., 1., 1., 1., 1., 1.]
```

**Vision Transformer**
```
PATCH_SIZE = 16
DEPTH = 8
HIDDEN_DIM = 256
K_DIM = 64
NUM_HEADS = 8

LR = 3e-4
STEP_SIZE = 5
GAMMA = 0.9
MAX_EPOCH = 200

DROPOUT = 0.1
EMB_DROPOUT = 0.1

class_weights = [1., 1., 1., 1., 1., 1., 1., 1.]
```
Train Acc: 82.61%\
Val Acc: 81.5%\
Test Acc: 81.66%

**ResNet-50**
```
BATCH_SIZE = 64

LR = 5e-5
STEP_SIZE = 10
GAMMA = 0.1
MAX_EPOCH = 200

class_weights = [1., 1., 1., 1., 1., 1., 1., 1.]
```

## References

Linformer implementation in Pytorch: https://github.com/lucidrains/linformer

Vision Transformer in Pytorch: https://github.com/lucidrains/vit-pytorch
