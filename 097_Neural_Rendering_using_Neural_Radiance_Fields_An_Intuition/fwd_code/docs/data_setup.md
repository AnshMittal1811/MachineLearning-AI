# Getting the Data

We use [ShapeNet](https://shapenet.org/) and [DTU MVS](https://roboimagedata.compute.dtu.dk) dataset in our experiments. 

## DTU Dataset

Same as [PixelNeRF](https://github.com/sxyu/pixel-nerf), we use DTU dataset with 4x downsampling. 
Also, we re-calculate the depth maps and camera parameters for rescaled images.
The processed data ```dtu_down_4.tar.gz``` could be found in [this link](https://drive.google.com/drive/u/1/folders/1-jggZm_kOLwi2ZS1LceAzh7Bn8QjkVfp).


## ShapeNet Dataset

For the ShapeNet experiments, we use the ShapeNet 64x64 dataset from [NMR](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip) (Hosted by DVR authors).
