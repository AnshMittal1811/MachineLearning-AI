# HV-plane reconstruction from a single 360 image

Code for our paper in CVPR 2021: **Indoor Panorama Planar 3D Reconstruction via Divide and Conquer** ([paper](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Indoor_Panorama_Planar_3D_Reconstruction_via_Divide_and_Conquer_CVPR_2021_paper.html), [video](https://www.youtube.com/watch?v=2uvP0V1oGRo))

![teaser](./static/teaser.png)


### Pretrained models
Download our pretrained models from [google drive](https://drive.google.com/drive/folders/10L2bayzVk7KOm2uhLCRkwh9P8zASE51i?usp=sharing) or [dropbox](https://www.dropbox.com/sh/v31g4hmxa6rlxd5/AADoOCW8FS34joCtr1pTjNr4a?dl=0).

### Inference on 360 datas
1. Please resize your images into `512 x 1024`.
2. Follow the preprocessing step [here](https://github.com/sunset1995/HorizonNet#1-pre-processing-align-camera-rotation-pose) to ensure Mahattan alignment of your 360 images.
3. Run our inference script. Examples:
```
python inference.py --pth ckpt/mp3d.pth --glob static/demo.png --outdir static/mp3d_model_results
```
To run on a batch of images, you can use `--glob "AWESOME_360_IMAGES_DIR/*png"`

### Visualize the results
Here is the visulization example on a held-out data:
```
python vis_planes.py --img static/demo.png --h_planes static/mp3d_model_results/demo.h_planes.exr --v_planes static/mp3d_model_results/demo.v_planes.exr --mesh
```
<img src="./static/example.png" width="300" />

To always visualize all the planes, add `--mesh_show_back_face`.

### Citation
```
@inproceedings{SunHWSC21,
  author    = {Cheng Sun and
               Chi{-}Wei Hsiao and
               Ning{-}Hsu Wang and
               Min Sun and
               Hwann{-}Tzong Chen},
  title     = {Indoor Panorama Planar 3D Reconstruction via Divide and Conquer},
  booktitle = {CVPR},
  year      = {2021},
}
```
