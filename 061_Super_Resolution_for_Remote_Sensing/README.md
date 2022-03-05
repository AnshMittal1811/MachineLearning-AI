## SRCDNet

The pytorch implementation for "[Super-resolution-based Change Detection Network with Stacked Attention Module for Images with Different Resolutions](https://ieeexplore.ieee.org/document/9472869) " on IEEE Transactions on Geoscience and Remote Sensing.  

The **SRCDNet** is designed to learn and predict change maps from bi-temporal images with different resolutions, which can be neatly turned into a **CDNet** and applied to images without any resolution difference.



## Requirements

- Python 3.6

- Pytorch 1.2.0

## Datasets

- Download the [BCDD Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)
- Download the [CDD Dataset](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)

Since the intial BCDD and CDD dataset only contains bi-temporal images with the same resolution, in our experiment, images of the latter phase are down-samplinged by 4 and 8 times to simulate resolution difference of 4 and 8 times (4x and 8x), respectively. 

The data folder is structured as follows:

```
├── data/   
│   ├── CDD/    # CDD dataset
|   |   ├── train/    # traning set 
|   |   |   ├── time1/    #images of time t1
|   |   |   ├── time2/    #images of time t2
|   |   |   ├── time2_lr/    #lower resolution images of time t2
|   |   |   |   ├── X4.00/    #4 times resolution difference
|   |   |   |   ├── X8.00/    #8 times resolution difference
|   |   |   ├── label/    #ground truth
|   |   ├── val/    # validation set, have the same structure of the training set 
│   ├── BCDD/    # BCDD dataset, have the same structure of the CDD dataset
│   └── 			
└── epochs/    # path to save the model
│   ├── CD/     
│   ├── SR/
│   └── 
...
```


## Train Examples 

- Train **SRCDNet** on **CDD** with **4x** resolution difference 

```
python train_srcd.py  --scale 4 
--hr1_train '../data/CDD/train/time1' 
--lr2_train '../data/CDD/train/time2_lr/X4.00'
--hr2_train '../data/CDD/train/time2'
--lab_train '../data/CDD/train/label'
--hr1_val '../data/CDD/val/time1'
--lr2_val '../data/CDD/val/time2_lr/X4.00'
--hr2_val '../data/CDD/val/time2'
--lab_val '../data/CDD/val/label' 
--model_dir 'epochs/X4.00/CD/'
--sr_dir 'epochs/X4.00/SR/'
--sta_dir 'statistics/CDD_4x.csv'
```

***Note** that more optional arguments could be found and retified in **configures.py**, including: num_epochs, gpu_id, batchsize, lr, etc. 



- Train **SRCDNet** on **BCDD** with **8x** resolution difference 

```
python train_srcd.py  --scale 8 
--hr1_train '../data/BCDD/train/time1' 
--lr2_train '../data/BCDD/train/time2_lr/X4.00'
--hr2_train '../data/BCDD/train/time2'
--lab_train '../data/BCDD/train/label'
--hr1_val '../data/BCDD/val/time1'
--lr2_val '../data/BCDD/val/time2_lr/X4.00'
--hr2_val '../data/BCDD/val/time2'
--lab_val '../data/BCDD/val/label' 
--model_dir 'epochs/X8.00/CD/'
--sr_dir 'epochs/X8.00/SR/'
--sta_dir 'statistics/BCDD_8x.csv'
```

- Train **CDNet** on **CDD** with **no** resolution difference 

```
python train_cd.py  
--hr1_train '../data/BCDD/train/time1' 
--hr2_train '../data/BCDD/train/time2'
--lab_train '../data/BCDD/train/label'
--hr1_val '../data/BCDD/val/time1'
--hr2_val '../data/BCDD/val/time2'
--lab_val '../data/BCDD/val/label' 
--model_dir 'epochs/X0.00/CD/'
--sta_dir 'statistics/CDD_0x.csv'
```



## Citation

Please cite our paper if you use this code in your work:

```
@ARTICLE{liu2021super,
  author={Liu, Mengxi and Shi, Qian and Marinoni, Andrea and He, Da and Liu, Xiaoping and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Super-Resolution-Based Change Detection Network With Stacked Attention Module for Images With Different Resolutions}, 
  year={2021},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2021.3091758}}
```



## Acknowledgment

This code is heavily borrowed from the [SRGAN](https://github.com/leftthomas/SRGAN) and [STANet](https://github.com/justchenhao/STANet).
