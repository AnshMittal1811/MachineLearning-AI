# What's in your hands? 3D Reconstruction of Generic Objects in Hands
Yufei Ye, Abhinav Gupta, Shubham Tulsiani
in CVPR 2022

Our work aims to reconstruct hand-held objects given a single RGB image. In contrast to prior works that typically assume known 3D templates and reduce the problem to 3D pose estimation, our work reconstructs generic hand-held object without knowing their 3D templates. 

[[Project Page]](https://judyye.github.io/ihoi) [[Video]](https://youtu.be/-hHlkWwENiI) [[Colab Demo]](https://colab.research.google.com/drive/1FdaBn4HQpf9p192CnEl25BQCxAzVfnzT?usp=sharing) [[Demo Code]](demo.ipynb) [[Arxiv]]() 



<img src=https://judyye.github.io/ihoi/ho3d/train_MDF10_0151.png width=200/> <img src=https://judyye.github.io/ihoi/rhoi/study_v_LDQPZ8ZeEec_frame000431.png width="200"/><img src="https://judyye.github.io/ihoi/rhoi/study_v_9LvWHppZVPU_frame000247.png" width="200"/>  

<img src="https://judyye.github.io/ihoi/teaser/drill.gif" width="200"/> <img src="https://judyye.github.io/ihoi/teaser/mouse.gif" width="200"/> <img src="https://judyye.github.io/ihoi/teaser/pen.gif" width="200"/> 
 

## Installation 
See [`install.md`](docs/install.md)

## Quick Start 

- Step by step [interactive notebook](demo.ipynb) 

- Or python script
    ```
    python -m demo.demo_image --filename demo/test.jpg --out output/ -e weights/mow/
    ```
We also provide some other images `docs/demo_%02d.jpg` for you to play around.

## Evaluation 

Coming soon
<!-- ```
python -m models.ihoi   --eval --ckpt PATH_TO_YOUR_CKPT/checkpoints/last.ckpt  

python -m models.ihoi   --eval --ckpt PATH_TO_YOUR_CKPT/checkpoints/last.ckpt  [--config experiments/[ho3d,mow].yaml  --slurm]

``` -->

## Train your own model

### Preprocess data
[`preprocess.md`](docs/preprocess.md) (Coming Soon)

### Start training
```
# obman
python -m models.ihoi --config experiments/obman.yaml  --slurm 

# finetune
python -m models.ihoi --config experiments/mow.yaml  --ckpt PATH_TO_OBMAN_MODEL/obman/checkpoints/last.ckpt --slurm

python -m models.ihoi --config experiments/ho3d.yaml  --ckpt PATH_TO_OBMAN_MODEL/obman/checkpoints/last.ckpt --slurm
```

## Citation
If you use find this code helpful, please consider citing:

```

@article{ye2022hand,
    author = {Ye, Yufei
              and Gupta, Abhinav
              and Tulsiani, Shubham},
    title = {What's in your hands? 3D Reconstruction of Generic Objects in Hands},
    booktitle = {CVPR},
    year={2022}
}
```

## TODO
- Demo:
    + [ ] support left hand
- preprocess:
    + [ ] provide cached data
    + [ ] how to create cached data
- eval:
    + [ ] test time refinement
    + [ ] predicted hand eval
    + [ ] add more demo images
    + [ ] models