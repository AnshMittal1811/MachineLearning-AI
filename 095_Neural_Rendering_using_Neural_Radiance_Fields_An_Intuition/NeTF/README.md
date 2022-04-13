# NeTF_public
The repository is the source code for paper "Non-line-of-Sight Imaging via Neural Transient Fields". [[Paper]](https://arxiv.org/abs/2101.00373#:~:text=Title%3ANon-line-of-Sight%20Imaging%20via%20Neural%20Transient%20Fields.%20Non-line-of-Sight%20Imaging,within%20a%20pre-defined%20volume%29%20of%20the%20hidden%20scene.)

The preprocessed data we use can be downloaded at [[Google Drive]](https://drive.google.com/file/d/1kGVrFcNZZbZs0ute_roEOg5UkYeh3jRl/view?usp=sharing) or [[Baidu Netdisk]](https://pan.baidu.com/s/16lWXwhm8CbXWAumJmlw9MQ) with password: netf

The raw data can be downloaded at [Zaragoza NLOS synthetic dataset](https://graphics.unizar.es/nlos_dataset.html), [f-k migration](http://www.computationalimaging.org/publications/nlos-fk/) and [Convolutional Approximations](https://imaging.cs.cmu.edu/conv_nlos/)

We also provide MATLAB code 'zaragoza_preprocess.m' and 'fkdata_preprocess.m' to convert data from Zaragoza dataset and fk to fit NeTF for those who want to run NeTF at other scene. 

# Environment setup
Make sure that the dependcies in `requirements.txt` are installed, or they can be installed by 
```
"pip install -r requirements.txt"
```

# How to run
Make sure that data is place correctly like
```
NeTF_public
│   README.md
│   run_netf.py
│   ...
│
└───data
    │   fk_dragon_meas_180_min_256_preprocessed.mat
    │   ...
    │
    └───zaragozadataset
        │   zaragoza256_preprocessed.mat
        │   ...
 
```
Then run with preset settings:
```
"python run_netf.py --config configs/zaragoza_bunny.txt"
```
Different settings are stroaged at "./configs/".

Under preset settings, the training process takes around 24 hours on a single NVIDIA Tesla M40 GPU.

# Results
The final volume and slices from different view are stroaged at "./model"

The matlab script "show_result.m" is also provided to generate 2D images from different views and 3D density distribution.

And the comparision between predicted and measured histogram is stroaged at "./figure"

# Contact us
Please email shensy@shanghaitech.edu.cn or wangzi@shanghaitech.edu.cn if you have any questions or suggestions.
