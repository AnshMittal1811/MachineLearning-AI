# UPST-NeRF: Universal Photorealistic Style Transfer of Neural Radiance Fields for 3D Scene

UPST-NeRF(see [our paper](https://arxiv.org/abs/2208.07059) and [project page](https://semchan.github.io/UPST_NeRF/) )is capable of rendering photorealistic stylized novel views with a consistent appearance at various view angles in 3D space.

<div align=center><img height="300" src="./imgs/teaser.png"/></div>

### Qualitative comparisons
<div align=center><img height="400" src="./imgs/llff_dataset_upst-nerf.gif"/></div>
<div align=center><img height="400" src="./imgs/nerf_dataset_upst-nerf.gif"/></div>

### Installation
```
git clone https://github.com/semchan/UPST-NeRF.git
cd UPST-NeRF
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent, please install the correct version for your machine.

<details>
  <summary> Dependencies (click to expand) </summary>

  - `PyTorch`, `numpy`, `torch_scatter`: main computation.
  - `scipy`, `lpips`: SSIM and LPIPS evaluation.
  - `tqdm`: progress bar.
  - `mmcv`: config system.
  - `opencv-python`: image processing.
  - `imageio`, `imageio-ffmpeg`: images and videos I/O.
</details>


## Download: datasets, trained models, and rendered test views

<details>
  <summary> Directory structure for the datasets (click to expand; only list used files) </summary>

    data
    ├── coco     # Link: http://cocodataset.org/#download
    │   └── [mscoco2017]
    │       ├── [train]
    │           └── r_*.png
       
    ├── nerf_synthetic     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    │   └── [chair|drums|ficus|hotdog|lego|materials|mic|ship]
    │       ├── [train|val|test]
    │       │   └── r_*.png
    │       └── transforms_[train|val|test].json
    │
    │
    └── nerf_llff_data     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
        └── [fern|flower|fortress|horns|leaves|orchids|room|trex]

</details>

### Synthetic-NeRF datasets
We use the datasets organized by [NeRF](https://github.com/bmild/nerf). Download links:
- [Synthetic-NeRF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (manually extract the `nerf_synthetic.zip` to `data/`)


### LLFF dataset
We use the LLFF dataset organized by NeRF. Download link: [nerf_llff_data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).




### Train
To train `fern` scene and evaluate testset `PSNR` at the end of training, run:
```bash
$ python run_upst.py  --config configs/llff/fern.py  --style_img ./style_images/your_image_name.jpg
```


### Evaluation
To only evaluate the trained `fern`, run:
```bash
$ python run_upst.py --config configs/llff/fern.py --style_img ./style_images/your_image_name.jpg --render_style --render_only --render_test --render_video
```

We also share some checkpoints for the 3D senes on llff dataset in baidu disk. You can download and put it into "./logs" for evaluation.

link：https://pan.baidu.com/s/1UxZqCiJIsL94jVonqadEFA 
code：5xk3 







## Acknowledgement
Thanks very much for the excellent work of DirectVoxGO, our code base is origined from an awesome [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) implementation.

