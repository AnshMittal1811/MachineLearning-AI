Video samples: [here](https://youtu.be/1ovewzz6u78) and [ablation](https://youtu.be/lzlAp05DhwQ)
# AutoNeRF

Report for this project [here](Praktikum_Report.pdf)


## Abstract
The goal of the practical was to come up with a generative model that is able to generate novel views of a 3d scene extremely quickly and is trained with very few images of that scene. Ultimately, we combined multiple generative models, namely neural radiance fields (NeRF), a variational autoencoder (VAE) and a conditional invertible neural network (cINN). We carefully combine them to get a generative model that fits these criteria. As a result we obtain a very quick inference of novel views with only very few training images.
We demonstrate, that the architecture can easily be modified to estimate the pose of the observer given an image.

## How to run
There are three notebooks provided, all of which are different parts of the pipeline
* nerf_train.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uprestel/AutoNeRF/blob/master/nerf_train.ipynb) This notebook trains the NeRF model and saves samples into a dataset.
* cinn_train.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uprestel/AutoNeRF/blob/master/cinn_train.ipynb)
This notebook provides the training for the VAE and the cINN. It also allows to show the samples and render videos of both NeRF and our model for comparison
* pose_train.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uprestel/AutoNeRF/blob/master/pose_estimation.ipynb) This notebook is the experimental part of this project, where we slightly modify the architecture to do pose estimation. We also gather error statistics in this notebook.
## Samples

![sample](/images/samples.png)

## Dataset
* lego https://drive.google.com/file/d/1Gb0AbE3KPkJYDJnNgYtZG58ntuJNpVjQ/view?usp=sharing
* chair https://drive.google.com/file/d/1kZoZyUoizT8ICnWBKt6OpBWdFHXdoICt/view?usp=sharing
* hotdog https://drive.google.com/file/d/1Chp2-2odW-leLXgF4_MG_7d69ZAJOKwN/view?usp=sharing
* (We use modified versions of the NeRF-datasets https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

