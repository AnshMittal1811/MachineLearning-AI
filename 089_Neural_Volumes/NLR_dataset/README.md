# Neural Lumigraph Rendering - Dataset

This is a public release of multi-view dataset published in the CVPR 2021 paper *Neural Lumigraph Rendering*.
The data are freely available for research purposes.
If you use this dataset, please cite:
```
@inproceedings{kellnhofer2021cvpr,
    author = {Petr Kellnhofer and Lars C. Jebe and Andrew Jones and Ryan Spicer and Kari Pulli and Gordon Wetzstein},
    title = {Neural Lumigraph Rendering},
    booktitle = {IEEE Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}
```

## Decription

The dataset consists of 7 folders each containing 1 multiview frame of a human bust captured by our multi-camera capture system. Please refer to our paper for details.
The data consist of:
- `rgb` - folder with sequentially indexed JPG images. The images have been rectified and color processing has been applied to remove the green spill from the green screen. The images 16-21 are high resolution central frames (3000x4000px) from our 6 high-resolution cameras.
- `mattes` - folder with correspding object masks matching the RGB frames.
- `calib_export.yaml` - YAML file with camera intrinsics and extrinsics (after rectification). The matrices are in a standard OpenCV format.

