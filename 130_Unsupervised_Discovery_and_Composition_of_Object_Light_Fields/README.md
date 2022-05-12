# Compositional Object Light Fields

[Project website](https://cameronosmith.github.io/colf)

Neural scene representations, both continuous and discrete, have recently emerged as a powerful new paradigm for 3D scene understanding. Recent efforts have tackled unsupervised discovery of object-centric neural scene representations. However, the high cost of ray-marching, exacerbated by the fact that each object representation has to be ray-marched separately, leads to insufficiently sampled radiance fields and thus, noisy renderings, poor framerates, and high memory and time complexity during training and rendering. Here, we propose to represent objects in an object-centric, compositional scene representation as light fields. We propose a novel light field compositor module that enables reconstructing the global light field from a set of object-centric light fields. Dubbed Compositional Object Light Fields (COLF), our method enables unsupervised learning of object-centric neural scene representations, state-of-the-art reconstruction and novel view synthesis performance on standard datasets, and rendering and training speeds at orders of magnitude faster than existing 3D approaches.

![](https://cameronosmith.github.io/img/thumbmanip.gif)

## Data and Models

The three room datasets (created by Koven Yu) are formatted and can be found [here](https://drive.google.com/drive/folders/14BCRAByUP7SIEBzgDxUROZ_t_vNJXglT?usp=sharing),
along with the corresponding pretrained models. 

## Training and Evaluation
Training and testing config files can be found in the folders train_configs and test_configs. 

For example, to evaluate on clevr, run `python test.py --config_filepath 
test_configs/clevr.yml`. 

See `python train.py --help` for all train options (and on test.py for test options).

Training summaries are written an 'events' folder in the logs directory specified by the config file or command-line arguments, to be viewed with tensorboard. Model checkpoints are saved in a 'checkpoints' folder relative to the same specified logs directory.

### Coordinate and camera parameter conventions
This code (based on Vincent Sitzmann's SRN and LFN code base) uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), 
the X-axis points right, and the Z-axis points into the image plane. Camera poses are assumed to be in a "camera2world" format,
i.e., they denote the matrix transform that transforms camera coordinates to world coordinates.

The code also reads an "intrinsics.txt" file from the dataset directory. This file is expected to be structured as follows (unnamed constants are unused):
```
f cx cy 0.
0. 0. 0.
1.
img_height img_width
```
The focal length, cx and cy are in pixels. Height and width are the resolution of the image.

## Acknowledgements
The code base is adapted from Vincent Sitzmann's [LFN](https://github.com/vsitzmann/light-field-networks).
Slot attention and CNN encoder code is from Koven Yu's [uORF](https://github.com/KovenYu/uORF/blob/main/README.md).
If you encounter any issues please email me at omid.smith.cameron@gmail.com or open an issue.
