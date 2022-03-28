# XAI-GAN

## Table of Contents
* Introduction
* Usage
* Report

## Introduction
Generative Adversarial Networks (GANs) are a revolutionary class of Deep Neural Networks (DNNs) that have been successfully used to generate realistic images, music, text, and other data. 
However, GAN training presents many challenges, notably it can be very resource-intensive. A potential weakness in GANs is that it requires a lot of data for successful training and data collection can be an expensive process. Typically, discriminator DNNs provide only one value (loss) of corrective feedback to generator DNNs (namely, the discriminator's assessment of the generated example). By contrast, we propose a new class of GAN we refer to as xAI-GAN that leverages recent advances in explainable AI (xAI) systems to provide a "richer" form of corrective feedback from discriminators to generators. 

### Source Code
The code of the project is found in the `/src/` directory and run using the main.py file. 

### Reproduce Results
To run any of the GANs specified (with or without xAI):
1. Go to `src/experiment_enums.py` and fill out the enums(s) of experiments to run (the file shows what to include)
2. Run `python3 main.py`
3. The results for each experiment (except for the FID scores) will be found under src/results
While the source code contains everything necessary to run the experiments, the code for evaluation is not ours. In order to evaluate (using FID scores), do the following:
4. For CIFAR10, clone `https://github.com/bioinf-jku/TTUR/blob/master/fid_example.py` and pass the folder of the generated images.
5. For MNIST or Fashion MNIST, clone `https://github.com/mseitzer/pytorch-fid` and pass the paths of numpy files generated.
6. In order to run Differentiable Augmentation technique, copy the `https://github.com/mit-han-lab/data-efficient-gans/DiffAugment_pytorch.py` file and follow the instructions in `https://github.com/mit-han-lab/data-efficient-gans`.
