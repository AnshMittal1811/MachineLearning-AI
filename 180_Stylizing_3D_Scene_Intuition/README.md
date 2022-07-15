# Stylizing-3D-Scene
PyTorch implementaton of our WACV 2022 paper "Stylizing 3D Scene via Implicit Representation and HyperNetwork".
You can visit our project website [here](https://ztex08010518.github.io/3dstyletransfer/).

In this work, we aim to address the 3D scene stylization problem - generating stylized images of the scene at arbitrary novel view angles.  
<div align=center><img height="230" src="https://github.com/ztex08010518/Stylizing-3D-Scene/blob/main/sample/teaser.png"/></div>

## Paper
[Stylizing 3D Scene via Implicit Representation and HyperNetwork](https://openaccess.thecvf.com/content/WACV2022/papers/Chiang_Stylizing_3D_Scene_via_Implicit_Representation_and_HyperNetwork_WACV_2022_paper.pdf)  
[Pei-Ze Chiang*](mailto:ztex080104518.cs08g@nctu.edu.tw), [Meng-Shiun Tsai*](mailto:infinitesky.cs08g@nctu.edu.tw), [Hung-Yu Tseng](https://hytseng0509.github.io/), [Wei-Sheng Lai](https://www.wslai.net/), [Wei-Chen Chiu](https://walonchiu.github.io/)  
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022.

Please cite our paper if you find it useful for your research.  
```
@InProceedings{Chiang_2022_WACV,
    author    = {Chiang, Pei-Ze and Tsai, Meng-Shiun and Tseng, Hung-Yu and Lai, Wei-Sheng and Chiu, Wei-Chen},
    title     = {Stylizing 3D Scene via Implicit Representation and HyperNetwork},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
}
```

## Environment Setting
* This code was developed with Python 3.6.12 & Pytorch 1.6.0 & CUDA 10.0

## Dataset and Pretrained Weight
* Dataset: [Tanks and Temples](https://drive.google.com/file/d/15-4XEjFf7YAOh2ft9RC_DZew11YMjNCj/view?usp=sharing) and [Wikiart images](https://www.kaggle.com/c/painter-by-numbers)  
NOTE: Download the datasets from here and unzip them under ./Tanks\_and\_Temples and ./wikiart/train.
* Pretrained Weights: [Second stage](https://drive.google.com/drive/folders/1hu7NSdi1NxgrxDxmkek3G_hNhytRrWoA?usp=sharing)  
NOTE: Download the pretrained weight from here and put it under ./logs/[scene\_name]\_second\_stage/ (scene\_name: Family, Francis, Horse, Playground, Truck)  

## Testing
1. First Stage: 
```
CUDA_VISIBLE_DEVICES=0,1 python ddp_test_nerf.py --config configs/test_family_second.txt --render_splits test
```
2. Second Stage:
```
bash test_script_second.sh
```

## Training
1. First Stage:
```
CUDA_VISIBLE_DEVICES=0,1 python ddp_train_nerf.py --config configs/train_family_first.txt
```
2. Second Stage:
```
CUDA_VISIBLE_DEVICES=0,1 python ddp_train_nerf.py --config configs/train_family_second.txt
```

## Acknowledgments
Our code is based on [NeRF++: Analyzing and Improving Neural Radiance Fields](https://github.com/Kai-46/nerfplusplus).  
The implementation of Hypternetwork and Style-VAE are based on [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://github.com/vsitzmann/scene-representation-networks) and [Adversarial Style Mining for One-Shot Unsupervised Domain Adaptation](https://github.com/RoyalVane/ASM).  
The implementation of Consistency metric(Temporal Warping Error) is borrowed from [Learning Blind Video Temporal Consistency](https://github.com/phoenix104104/fast_blind_video_consistency).
