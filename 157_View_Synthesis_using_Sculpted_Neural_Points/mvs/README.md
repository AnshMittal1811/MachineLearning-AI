# Run SNP on Your Own Scenes

For training on your own scene, the easiest way is probably to reuse the data loading and processing code for LLFF, since the LLFF input is just some (unposed) 2D images.

We here use the `fern` scene as an example on how to get a pose and a dense depth map for each input image. You will need to replace `fern` with your own scene name.

## 1. install additional dependencies

### 1.1 Install CER-MVS
We use a (slightly customized) [CER-MVS](https://github.com/princeton-vl/CER-MVS) network. Install the following dependencies in the `snp` enviroment in order to run the MVS network.

```
conda activate snp
conda install opt_einsum
cd alt_cuda_corr && python setup.py install && cd ..
```

### 1.2 Install COLMAP.
We use the COLMAP customized by the [NerfingMVS](https://github.com/weiyithu/NerfingMVS) authors, which added a fusion mask output. Please follow the instructions in the [link](https://github.com/B1ueber2y/colmap/tree/c84269d693246d8294307cc32f851813f18b6a2d) to install COLMAP.


## 2. Preprocess the raw images
Put your images under `./raw_images/fern/images` and run the following command:

```
python img_move_and_reisze.py --s ./raw_images/ --t ./raw_images_resized --h 1200 --w 1600
```

COLMAP can be slow, and therefore we resize the image to speed up COLMAP and avoid potential OOM error.

Note: we have only tested the pipeline under aspect ratio 4:3. If your image has other aspect ratio, we recommend resizing them into 4:3 in this step and resize the rendered images back to the original aspect ratio.

## 3. Extract camera poses and raw depth maps using COLMAP

Run 
```
sh colmap.sh <dir/to/your/colmap/colmap.bat> ./raw_images_resized/fern
```
You can check the validity of the COLMAP poses and depths by opening `./raw_images_resized/fern/dense/fused.ply` in some 3D viewer, e.g., [MeshLab](https://www.meshlab.net/):

![](https://github.com/princeton-vl/SNP/blob/main/figs/colmap_pt_fern_vis.png)

The COLMAP depths are often very noisy and incomplete, so next we run the MVS network to refine the raw depth maps.

## 4. Extract the COLMAP ouput into DTU format

DTU format is used by our dataloader. Run

```
python colmap2dtu.py --scene fern --colmap_output_dir ./raw_images_resized
```

## 5. Get the refined depth maps by finetune the MVS network

Download the pretrained MVS network weights [here](https://drive.google.com/drive/folders/189nUV9_9YM_0bLW1Y97SQ1nK_EVpxGW6?usp=sharing) and put it under `./checkpoints/DTU_trained.pth`

Run

```
sh depth_ft_inf.sh fern
```
The depth maps are saved as `.pfm` files under ./mvs_depths/ft_depths/fern/depths. You can also visualize the output by opening `./mvs_depths/ft_depths_s1/fern/depths/pctld.ply`:

![](https://github.com/princeton-vl/SNP/blob/main/figs/mvs_pt_fern_vis.png)


## 6. Rearrange the output files

Now you can treat your own scene as an extra LLFF scene. For example, you can move the output poses and depths into the main data folder and run the training script following the instructions in the main [README](https://github.com/princeton-vl/SNP):

```
mv ./raw_images_resized/fern ../data/LLFF/raw
mv ./mvs_depths/ft_depths/fern ../data/LLFF/depths
```

# Citation
If you use the CER-MVS network to reconstruct your scene, please consider also citing
```
@article{ma2022multiview,
  title={Multiview Stereo with Cascaded Epipolar RAFT},
  author={Zeyu Ma and Zachary Teed and Jia Deng},
  journal={arXiv preprint arXiv:2205.04502},
  year={2022}
}
```