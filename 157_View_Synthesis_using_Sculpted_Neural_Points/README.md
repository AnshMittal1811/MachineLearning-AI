# View Synthesis with Sculpted Neural Points

Official code for [View Synthesis with Sculpted Neural Points](https://arxiv.org/abs/2205.05869)

[![](https://github.com/princeton-vl/SNP/blob/main/figs/youtube_thumbnail.png)](https://www.youtube.com/watch?v=ctPBhvgVOow)
![](https://github.com/princeton-vl/SNP/blob/main/figs/fig1.png)




# Setup
We recommend installing the dependencies using Anaconda. Run the following commands to create a new env:
```
conda create -n snp python=3.8
conda activate snp
conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
conda install -c conda-forge plyfile imageio imageio-ffmpeg matplotlib tensorboard ffmpeg=4.3
pip install opencv-python==4.5.3.56 lpips==0.1.4 moviepy
```

Note: you will need a NVIDIA GPU with at least 11 GB memory to run the pre-trained checkpoints on LLFF and 48GB memory (we use A6000) to fully reproduce our results in the paper.

# Prepare the Datasets
You can find the pre-processed DTU and LLFF datasets in the [Google Drive link](https://drive.google.com/drive/folders/189nUV9_9YM_0bLW1Y97SQ1nK_EVpxGW6?usp=sharing).

Put the raw data and depths under `./data`.

The datasets should have the following structure:
```                                                                                           
├── data                                                                                                                                                                                                       
│   ├── DTU                                                                                                  
│   │   └── depths                                                                                                                            
│   │   └── raw                                                                            
|   ├── LLFF
|   |   └── depths                                                                                                                            
│   │   └── raw
```

# Evaluate the Pre-trained Models
The easiest way to get started is to try our pre-trained checkpoints.

You can find the saved checkpoints/pointclouds in the [Google Drive link](https://drive.google.com/drive/folders/189nUV9_9YM_0bLW1Y97SQ1nK_EVpxGW6?usp=sharing).

Put the downloaded `*.pth` and `*.pt` under `./saved_checkpoints` and `./saved_pointclouds`, respectively.

The checkpoints' file size could be large. Therefore, you can download the checkpoints on each individual scenes separately, but please put them following this structure:

```                                                                                           
├── saved_checkpoints                                                                                                                                                                                                      
│   ├── DTU                                                                                                  
│   │   └── ..._scan1_....pth   # checkpoints for the scan1 scene 
|   |   └── ...                                                                        
|   ├── LLFF
|   |   └── ..._fern_....pth   # checkpoints for the fern scene 
|   |   └── ...  
|
├── saved_pointclouds                                                                                                                                                                                                      
│   ├── DTU                                                                                                  
│   │   └── ..._scan1_....pt   # sculpted pointclouds for the scan1 scene 
|   |   └── ...                                                                        
|   ├── LLFF
|   |   └── ..._fern_....pt   # sculpted pointclouds for the fern scene 
|   |   └── ...  
```

Then run the following command:
```
sh eval.sh <dataset_name> <scene_name>
```
Where `<dataset_name>` is `DTUHR` for the DTU dataset and `LLFF` for the LLFF dataset.

For example, if you run 
```
sh eval.sh LLFF fern
```
you will find this GIF in `./saved_videos` and the evaluation numbers are printed in the command line.

![](https://github.com/princeton-vl/SNP/blob/main/figs/fern.gif)

# Train from Scratch
Run
```
sh train.sh <dataset_name> <scene_name>
```
to train the whole pipeline (including point sculpting and the final training). It takes about 14 hours on a single A6000 GPU.

You can find the tensorboard logs under `./tb` folder. The final checkpoints and pointclouds will be stored under `./checkpoints` and `./pointclouds`, respectively.

Note: for people who don't have GPUs with 48GB memory and/or want higher rendering speed, there are a few hyperparameters that you can play with. From our experience, changing these values (in a reasonable range) has a minor effect on the rendered image quality.

- Reduce `max_num_pts` which controls the point cloud size.
- Adjust `crop_h` and `crop_w` to control the image resolution.

# Run SNP on Your Own Scenes
See the [link](mvs/README.md) here.

# Citation
If you find this repository useful, please cite
```
@article{zuo2022view,
  title={View Synthesis with Sculpted Neural Points},
  author={Yiming Zuo and Jia Deng},
  journal={arXiv preprint arXiv:2205.05869},
  year={2022}
}
```