# SADRNet
Paper link: [SADRNet: Self-Aligned Dual Face Regression Networks for Robust 3D Dense Face Alignment and Reconstruction](https://arxiv.org/abs/2106.03021)

![image](https://github.com/MCG-NJU/SADRNet/blob/main/data/output/30904b202ac883bc14e59c1225c9316c.gif)
## Requirements
```
python                 3.6.2
matplotlib             3.1.1  
Cython                 0.29.13
numba                  0.45.1
numpy                  1.16.0   
opencv-python          4.1.1
Pillow                 6.1.0                 
pyrender               0.1.33                
scikit-image           0.15.0                
scipy                  1.3.1
torch                  1.2.0                 
torchvision            0.4.0
```

## Pretrained model

Link: [https://drive.google.com/file/d/1mqdBdVzC9myTWImkevQIn-AuBrVEix18/view?usp=sharing](https://drive.google.com/file/d/1mqdBdVzC9myTWImkevQIn-AuBrVEix18/view?usp=sharing) .

Please put it under ```data/saved_model/SADRNv2/```.

Please set ```./SADRN``` as the working directory when running codes in this repo.

## Predicting

* Put images under ```data/example/```.

* Run ```src/run/predict.py```.

The network takes cropped-out 256×256×3 images as the input.

## Training

* Download 300W-LP and AFLW2000-3D at [http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm) .

* Extract them into ```'data/packs/AFLW2000'``` and ```'data/packs/300W_LP'```

* Please refer to [face3d](https://github.com/YadiraF/face3d/blob/master/examples/Data/BFM/readme.md) to prepare BFM data. And move the generated files in ```Out/``` to ```data/Out/``` 

* Run ```src/run/prepare_dataset.py```, it will take several hours.

* Run ```train_block_data.py```.  Some training settings are included in ```config.py``` and ```src/configs```.

## Acknowledgements
We especially thank the contributors of the [face3d](https://github.com/YadiraF/face3d/blob/master/examples/Data/BFM/readme.md) codebase for providing helpful code.
