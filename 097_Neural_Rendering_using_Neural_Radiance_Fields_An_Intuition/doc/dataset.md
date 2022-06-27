# Dataset
In the following, we provide file structure, camera coordinate system, data generation details for two datasets: Replica, Tanks&Temple.
## 📷 Camera Coordinate System 
In all datasets, the camera coordinate system follows that in project: [NeRF in PyTorch3D](https://github.com/facebookresearch/pytorch3d/tree/main/projects/nerf), which follows the PyTorch3D convention. The figure below illustrate the axis orientation in each coordinate system, please refer to PyTorch3D [official document](https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md) for configuration and usage details.

![camera_model](camera_model.png)
## 🛋️ Replica
### Data Generation
The Replica dataset is composed of several indoor scenes, and we generate data of 7 distint scenes for training and evaluation. First of all, we download Replica Dataset from the [official repo](https://github.com/facebookresearch/Replica-Dataset)$^1$. The camera views, RGB images, depth map are sampled with rendering engine [BlenderProc](https://github.com/DLR-RM/BlenderProc)$^2$, which provides an interactive viewing interface and data generation pipeline. The figure below illustrates the camera poses distribution. For each scene, we sample 50 training views (<font color=blue>blue dots</font>) and reconstruct point cloud with [COLMAP](https://colmap.github.io/) $^{3,4}$. In addition, we sample another 100 novel views (<font color=red>red dots</font>) of much wider viewing range for evaluation. As a result, the data is able to validate the extrapolation capability.
 
![replica_camera](replica_camera.png)
### Format
```bash
<scene>/<split>      # split=train/valid
|- dense
    |- points3D.txt  # point cloud reconstructed from training views
|- images            # RGB images
    |- 00000.jpg
    |- 00001.jpg
    |- 00002.jpg
    ...
|- depth.npy         # depth maps
|- R.npy             # camera extrinsics: rotation
|- T.npy             # camera extrinsics: translation
```
## 👨‍👩‍👦 Tanks&Temple
### Data Generation
We download dataset from project [NSVF](https://github.com/facebookresearch/NSVF) $^5$, and transform the camera coordinate system to PyTorch3D convention. Besides, we reconstruct point cloud from training views with [COLMAP](https://colmap.github.io/) $^{3,4}$. Below, we provide a python script for conversion (NSVF to PyTorch3D).
```bash
python dataset_convert.py --dataset_dir [directory of NSVF dataset] --output_dir [directory of converted dataset] --img_w [width] --img_h [height]
# example: python dataset_convert.py --dataset_dir TanksAndTemple/ --output_dir convert/TanksAndTemple --img_w 1920 --img_h 1080
```
### Format
```bash 
<scene>/<split>    # split=train/valid
|- images          # RGB images
    |- *.png
|- cameras.txt     # Camera intrinsics: image size, focal length, principle point
|- points3D.npy    # point cloud reconstructed from training views
|- R.npy           # camera extrinsics: rotation
|- T.npy           # camera extrinsics: translation

```
---
#### References
1. *The Replica Dataset: A Digital Replica of Indoor Spaces*, in ArXiv, 2019
2. *BlenderProc*, in ArXiv, 2019
3. *Structure-from-Motion Revisited*, in CVPR, 2016
4. *Pixelwise View Selection for Unstructured Multi-View Stereo*, in ECCV, 2016
5. *Neural Sparse Voxel Fields*, in NeurIPS, 2020