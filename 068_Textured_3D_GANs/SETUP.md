# Code and data setup

## Requirements
- [Kaolin v0.1](https://github.com/NVIDIAGameWorks/kaolin/tree/v0.1) (the newer version v0.9 is also supported, but see the remarks below)
- Python >= 3.6
- PyTorch >= 1.6
- CUDA >= 10.0
- Misc dependencies (installable using pip): `packaging`, `tensorboard` (optional).

You can install Kaolin v0.1 as follows:
```
git clone https://github.com/NVIDIAGameWorks/kaolin --branch v0.1 --single-branch
cd kaolin
python setup.py install
```
**Note on newer Kaolin versions:** for convenience, we have added support for Kaolin v0.9. This new version should work as a drop-in replacement of the previous version, but note that all our models were trained using the older Kaolin v0.1. Therefore, if your goal is to replicate our results, you should use v0.1. We have nonetheless observed that, at least on inference code, the results between the two versions are equal and v0.9 is slightly faster.

Optionally, if you want to want to infer segmentations from scratch or use a custom dataset, you should also install:
- [Detectron2](https://github.com/facebookresearch/detectron2) >= 0.4
- [seg_every_thing](https://github.com/ronghanghu/seg_every_thing) (requires a Python 2.7 distribution since it builds upon the older Detectron 1)

For the datasets used in our work (ImageNet, CUB, Pascal3D+) we provide pre-computed segmentation masks and part detections, so you do not need to set up the two dependencies above. 

## Minimal setup (evaluating pretrained models)
With this setup you will be able to evaluate our pretrained models (FID), generate qualitative demos, and export mesh samples that can be loaded into modelling tools (e.g. Blender), but you will not be able to train a new model from scratch. 

It suffices to download the pretrained models and cache directory from the [Releases](https://github.com/dariopavllo/textured-3d-gan/releases) section, and extract the archives to the root directory of this repo. You do not need to set up any dataset.

## Full dataset setup (training from scratch)
If you have not already done so, set up the cache directory as described in the step above.

We then provide instructions on how to set up the various datasets used in our approach:
- **[ImageNet22k](#ImageNet22k):** motorbike, bus, truck, car, airplane, sheep, elephant, zebra, horse, cow, bear, giraffe
- **[Pascal3D+](#Pascal3D+):** car, airplane (subsets of ImageNet, primarily used for comparing to previous work)
- **[CUB](#CUB):** bird

You only need to perform these steps if you intend to train a new model from scratch. Evaluating pretrained models does not require any dataset setup.
Additionally, each dataset is optional, e.g. if you only want to train/evaluate on Pascal3D+ or CUB, you don't need to set up ImageNet.

## ImageNet22k
We adopt ImageNet22k instead of the more common ImageNet1k because some of our required classes/synsets are not available in ImageNet1k. You are of course free to use ImageNet1k for experimentation purposes, but you need to set up ImageNet22k to reproduce the results of our paper. You can download it from [Academic Torrents](https://academictorrents.com/details/564a77c1e1119da199ff32622a1609431b9f1c47). For reference, the table below shows the list of synsets required for each class:

| Category        | Synsets                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| motorcycle      | n03790512, n03791053, n04466871                                                                   |
| bus             | n04146614, n02924116                                                                              |
| truck           | n03345487, n03417042, n03796401                                                                   |
| car             | n02814533, n02958343, n03498781, n03770085, n03770679, n03930630, n04037443, n04166281, n04285965 |
| airplane        | n02690373, n02691156, n03335030, n04012084                                                        |
| sheep           | n10588074, n02411705, n02413050, n02412210                                                        |
| elephant        | n02504013, n02504458                                                                              |
| zebra           | n02391049, n02391234, n02391373, n02391508                                                        |
| horse           | n02381460, n02374451                                                                              |
| cow             | n01887787, n02402425                                                                              |
| bear            | n02132136, n02133161, n02131653, n02134084                                                        |
| giraffe         | n02439033                                                                                         |

To set up ImageNet, run the following command:
```
python setup_imagenet.py /path/to/imagenet/
```
where `/path/to/imagenet/` is your ImageNet folder. The script will autodetect the format (tar archives or extracted folders) and extract/copy all required files. If the archives are already extracted into folders and you wish to create symbolic links to them, you can specify `--symlinks`.

## Pascal3D+
Download [PASCAL3D+_release1.1.zip](ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip) ([source](https://cvgl.stanford.edu/projects/pascal3d.html)) and set up your directory tree like this:
```
datasets/p3d/PASCAL3D+_release1.1/
datasets/p3d/data/
datasets/p3d/sfm/
```

Creating a symbolic link to `PASCAL3D+_release1.1`  is also a good idea if you have a copy of the dataset somewhere else.

## CUB
Download [CUB images](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and [segmentations](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz) ([source](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)) and extract them so that your directory tree looks like this:
```
datasets/cub/CUB_200_2011/
datasets/cub/CUB_200_2011/segmentations/
datasets/cub/data/
datasets/cub/sfm/
```

# Custom datasets
To add a custom dataset, you need to modify `data/definitions.py` and add the mapping between the name of the dataset and the category. This mapping is used to retrieve the correct mesh template.

```
dataset_to_class_name['your_custom_dataset_name'] = 'category_name'
```

The mesh templates (in .obj format) are stored in `mesh_templates/classes/` and must be named by appending an incremental suffix, e.g. `car1.obj`, `car2.obj`, `car3.obj`. These templates must subsequently be remeshed as described in [TRAINING.md](TRAINING.md) (already done for the meshes used in our work). If your dataset represents an existing category (e.g. *car*), you are of course free to use the existing templates (in this case, no remeshing is required as the remeshed versions are already provided in the cache directory).

You can then store your images under `datasets/your_dataset_name/images/` and follow the next section for instructions on how to set up the detections.

# Computing detections from scratch
This step is not required for the datasets used in our work (ImageNet, Pascal3D+, CUB), as we provide precomputed detections in our cache directory. However, if you wish to re-compute them for replication reasons or because you are using a custom dataset, follow the instructions below.

For the object segmentation mask, we use [PointRend](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend), a state-of-the-art instance segmentation model which builds upon Detectron 2. For part segmentations, we use [seg_every_thing](https://github.com/ronghanghu/seg_every_thing), which requires a Python 2.7 distribution.

First, set up *seg_every_thing* by following the instructions on [their repository](https://github.com/ronghanghu/seg_every_thing). Next, copy our script `tools/detection_tool_vg3k.py` to the `tools` directory of their repo and run the following command from the root of their repo:
```
python2 tools/detection_tool_vg3k.py --output-dir /path/to/datasets/your_dataset_name/vg3k_detections /path/to/datasets/your_dataset_name/images
```

Finally, you can finalize the dataset/detections by running `detection_tool.py` from the root of this repo. By default, the tool processes all datasets, but you can also specify individual datasets or multiple comma-separated datasets, e.g.
```
python detection_tool.py [--datasets all]
python detection_tool.py --datasets imagenet
python detection_tool.py --datasets p3d
python detection_tool.py --datasets cub
python detection_tool.py --datasets p3d,cub
python detection_tool.py --datasets your_custom_dataset
```

This script will detect object masks using PointRend and discard images that don't pass the quality checks. The results/metadata will be saved to `cache/dataset_name/detections.npy`.


