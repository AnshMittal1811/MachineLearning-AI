### cosmoGAN

This code is to accompany "Creating Virtual Universes Using Generative Adversarial Networks" manuscript [arXiv:1706.02390](https://arxiv.org/abs/1706.02390).
The architecture is an implementation of the DCGAN architecture ([arXiv:1511.06434](https://arxiv.org/abs/1511.06434)).

- - - 
### How to train:  
```bash
git clone git@github.com:MustafaMustafa/cosmoGAN.git
cd cosmoGAN/networks
mkdir data
wget http://portal.nersc.gov/project/dasrepo/cosmogan/cosmogan_maps_256_8k_1.npy
cd ../
```

That will download sample data (8k maps) for testing. You can download more data from [here](http://portal.nersc.gov/project/dasrepo/cosmogan/). All of this data has been generated using our GAN and can be used to train your own. Original data is available upon request from the authors.

To run:
```bash
python run_dcgan.py
```


### Load pre-trained weights:  
First download the weights:
```bash
cd cosmoGAN/networks
wget http://portal.nersc.gov/project/dasrepo/cosmogan/cosmoGAN_pretrained_weights.tar
tar -xvf cosmoGAN_pretrained_weights.tar
```

Then take a look at [networks/load_and_use_pretrained_weights.ipynb](networks/load_and_use_pretrained_weights.ipynb) notebook for how to run.  
