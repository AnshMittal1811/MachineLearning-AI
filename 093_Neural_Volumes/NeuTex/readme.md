# NeuTex: Neural Texture Mapping for Volumetric Neural Rendering

Paper: [https://arxiv.org/abs/2103.00762](https://arxiv.org/abs/2103.00762) 

## Running
### Run on the provided DTU scene
```bash
cd run
bash dtu.sh 114
```
(Install any missing library from pip)

Further fine tuning for texture after fixing the geometry
```bash
bash dtu-freese.sh 114
```

### Run on custom datasets
Similar to the provided DTU scene, you will need to provide a custom data loader
similar to `data/dtu_dataset.py` and modify the dataset arguments in the bash
scripts accordingly.

Similar to the `dtu_dataset.py`, the custom dataset needs to provide the
following fields when getting and item:
- `gt_mask`, a 0/1 mask for background/foreground.
- `near`, the near plane for point sampling on the ray
- `far`, the far plane for point sampling on the ray
- `raydir`, ray directions
- `gt_image`, ground truth pixel colors
- `background_color`, color of the image background

The captured scene must be contained in the unit cube centered at world origin.

## Citation
```
@InProceedings{xiang2021neutex,
author = {Xiang, Fanbo and Xu, Zexiang and Hašan, Miloš and Hold-Geoffroy, Yannick and Sunkavalli, Kalyan and Su, Hao},
title = {{N}eu{T}ex: {N}eural {T}exture {M}apping for {V}olumetric {N}eural {R}endering},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2021}}
```
