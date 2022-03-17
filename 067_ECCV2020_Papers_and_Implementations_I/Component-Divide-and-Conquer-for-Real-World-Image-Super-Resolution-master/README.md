# Component Divide-and-Conquer for Real-World Image Super-Resolution(CDC)

This repository is an official PyTorch implementation of the paper **"Component Divide-and-Conquer for Real-World Image Super-Resolution
"** from **ECCV 2020**. [[PDF](http://arxiv.org/abs/2008.01928)]

We provide full training and testing codes, pre-trained models and the large-scale dataset used in our paper. You can train your model from scratch, or use a pre-trained model to enlarge your images.

## Code
### Dependencies
* Python 3.6
* PyTorch >= 1.1.0
* numpy
* cv2
* skimage
* tqdm

### Quick Start
Clone this github repo.
```bash
git clone https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution
cd Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution/CDC
```
#### Training
1. Download our dataset and unpack them to any place you want. Then, change the ```dataroot``` and ```test_dataroot``` argument in ```./options/realSR_HGSR_MSHR.py``` to the place where images are located.
2. Run ```CDC_train_test.py``` using script file ```train_pc.sh```.
```bash
sh ./train_pc.sh cdc_x4 ./CDC_train_test.py ./options/realSR_HGSR_MSHR.py 1
```
3. You can find the results in ```./experiments/CDC-X4``` if the ```exp_name``` argument in ```./options/realSR_HGSR_MSHR.py``` is ```CDC-X4```

#### Testing
1. Download our pre-trained models to ```./models``` folder or use your pre-trained models
2. Change the ```test_dataroot``` argument in ```CDC_test.py``` to the place where images are located
3. Run ```CDC_test.py``` using script file ```test_models_pc.sh```.
```bash
sh test_models_pc.sh cdc_x4_test ./CDC_test.py ./models/HGSR-MHR_X4_SubRegion_GW_283.pth 1
```
4. You can find the enlarged images in ```./results``` folder

### Pretrained models
1. [2X Models](https://drive.google.com/file/d/1GGcnUCGaBWStxh-78PnDIlaCfPMfATaG/view?usp=sharing)
2. [3X Models](https://drive.google.com/file/d/1VhppmVr159dlXzbVPh0zDcVGBblj2k0j/view?usp=sharing)
3. [4X Models](https://drive.google.com/file/d/18Bg1B5XvksMNsM1KXoPegsOhIbP6WnC4/view?usp=sharing)

The above provided models are both trained on our dataset with our gradient-weighted loss.

## Dataset
Please download our dataset from [Google Drive](https://drive.google.com/drive/folders/1tP5m4k1_shFT6Dcw31XV8cWHtblGmbOk?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1ey9JF4S5wLnE5Iw5z67R8A). The verification code is ```osiy```. There are 31970 192×192 patches cropped for training and 93 image pairs for testing.

 |Methods    |  Scale  |    PSNR    |    SSIM    |    LPIPS    |
 |-----------|---------|:----------:|:----------:|:-----------:|
 |Bicubic    |    2    |    32.67   |    0.887   |    0.201    |
 |EDSR       |    2    |    34.24   |    0.908   |    0.155    |
 |RCAN       |    2    |    34.34   |    0.908   |    0.158    |
 |CDC(ours)  |    2    |  **34.45** |  **0.910** |  **0.146**  |
 |Bicubic    |    3    |    31.50   |    0.835   |    0.362    |
 |EDSR       |    3    |    32.93   |    0.876   |    0.241    |
 |RCAN       |    3    |    33.03   |    0.876   |  **0.241**  |
 |CDC(ours)  |    3    |  **33.06** |  **0.876** |    0.244    |
 |Bicubic    |    4    |    30.56   |    0.820   |    0.438    |
 |EDSR       |    4    |    32.03   |    0.855   |    0.307    |
 |RCAN       |    4    |    31.85   |    0.857   |    0.305    |
 |CDC(ours)  |    4    |  **32.42** |  **0.861** |  **0.300**  |

## Citation
If you find our work useful in your research or publication, please cite:
```
@InProceedings{wei2020cdc,
  author = {Pengxu Wei, Ziwei Xie, Hannan Lu, ZongYuan Zhan, Qixiang Ye, Wangmeng Zuo, Liang Lin},
  title = {Component Divide-and-Conquer for Real-World Image Super-Resolution},
  booktitle = {Proceedings of the European Conference on Computer Vision},
  year = {2020}
}
```
