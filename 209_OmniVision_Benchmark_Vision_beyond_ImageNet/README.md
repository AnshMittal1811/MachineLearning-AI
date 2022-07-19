# [ECCV2022] OmniBenchmark

![teaser](./figures/paper_teaser-min.png)

<div align="center">

<div>
    <a href='https://zhangyuanhan-ai.github.io/' target='_blank'>Yuanhan Zhang</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=ngPR1dIAAAAJ&hl=zh-CN' target='_blank'>Zhenfei Yin</a><sup>2</sup>&emsp;
    <a href='https://amandajshao.github.io/' target='_blank'>Jing Shao</a><sup>2</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwe Liu</a><sup>1</sup>
</div>
<div>
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    <sup>2</sup>SenseTime Research&emsp;
</div>

---

<div>
    <a href='https://arxiv.org/abs/2207.07106' target='_blank'>[Paper]</a> 
    •
    <a href='https://zhangyuanhan-ai.github.io/OmniBenchmark' target='_blank'>[Project Page]</a>
    •
    <a href='https://paperswithcode.com/sota/image-classification-on-omnibenchmark' target='_blank'>[Leaderboard]</a>
    <br>
    <a href='https://codalab.lisn.upsaclay.fr/competitions/6043' target='_blank'>[Challenge:ImageNet1k-Pretrain Track]</a>
    •
    <a href='https://codalab.lisn.upsaclay.fr/competitions/6045' target='_blank'>[Challenge:Open-Pretrain Track]</a>
</div>
</div>

## Updates
[07/2022] OmniBenchmark Challenge ECCV@2022 will start together with [ECCV 2022 SenseHuman Workshop](https://sense-human.github.io/).

[07/2022] Dataset with hidden test has been released.

[07/2022] Code for ReCo has been released.

[07/2022] [arXiv](https://github.com/ZhangYuanhan-AI/OmniBenchmark) paper has been released.


## About OmniBenchmark
### Download data and annotations
```
cd download_tool
#it may cost 2 hours
python download_image.py
```
After downlaoding you should see the following folder structure, i.e., a separate folder of images per realm: 

```
<meta>
...
|--- activity
|   |--- activity.train
|   |   |---images/ #data
|   |   |    |---*.jpg
|   |   |---record.txt #annotation
|   |--- activity.val
|   |   |images/ #data
|   |   |    |---*.jpg
|   |   |--- record.txt #annotation
|   |--- activity.test
|   |   |images/ #data
|   |   |    |---*.jpg
|   |   |--- record.txt #image_path + pseudo_class
...
```
Please refer to ``download_tool/README.txt`` for the detail information of your downloaded files.

**IMPORTANT:** Each realm dataset has a train/val/test set, the annotation of the test is hidden *currently* for OmniBenchmark Challenge @ ECCV2022 and will be public after then. The result in the paper is evaluated on the test+val set.

### Find the class name 
In downloaded meta files (e.g. car.val), each line of the file is a data record, including the local image path and the corresponding label, separated by a space.
```
#path trainid
XXXXXX 0
XXXXXX 1
XXXXXX 2
...
``` 
You can find the name of ``trainid`` through ``trainid2name.json``. 


## Evaluating a model on the OmniBenchmark

### Step1: Model preparation
#### Public models
Inspired by [ImageNet-CoG](https://europe.naverlabs.com/research/computer-vision/cog-benchmark/), we use ResNet50 as a reference model, and evaluate 22 models that are divided into three groups. You can download these models at [HERE](https://drive.google.com/drive/folders/1zJcWHWK6olLPX44t4yE8WyM2Bq1jenAR?usp=sharing). You can check the reference papers of these model in the paper.

After you download models, you should update their path in their config files in the ``linear_probe/model_cfg/``.

e.g.
if you download beit_b16 model in the ./weights/beit_base_patch16_224_pt22k_ft22kto1k.pth
- ``vim linear_probe/model_cfg/beit_b16.yaml``
- Change ``/mnt/lustre/zhangyuanhan/architech/beit_base_patch16_224_pt22k_ft22kto1k.pth`` to ``./weights/beit_base_patch16_224_pt22k_ft22kto1k.pth``.

#### Customer models
- Upload your model files in ``linear_probe/models/ABC.config``, ABC is your model name.
- Upload the corresponding config files in ``linear_probe/configs/model_cfg/``.


### Step2: Data preparation
Updating the path of your downloaded data and annotation in ``linear_probe/configs/100p/``.

e.g. add the information of activity dataset.
- ``vim linear_probe/100p/config_activity.yaml``
- Update the ``root`` in line 13/19 and ``meta`` in line 14/20

### Step3: Linear probing
- ``vim linear_probe/multi_run_100p.sh``
- Change ``models=(beit_b16 effenetb4)`` to ``models=(beit_b16 effenetb4 ABC)``. Separating each model name in space. 
- Change ``datasets=(activity aircraft)`` to ``datasets=(activity aircraft DEF GHI)``. DEF and GHI is the dataset name you want to evaluate, refered to ``linear_probe/configs/100p/config_DEF.yaml``.
- ``sh linear_probe/multi_run_100p.sh``

## Citation
If you use this code in your research, please kindly cite this work.
```
@inproceedings{zhang2022omnibenchmark,
      title={Benchmarking Omni-Vision Representation through the Lens of Visual Realms}, 
      author={Yuanhan Zhang and Zhenfei Yin and Jing Shao and Ziwei Liu},
      year={2022},
      archivePrefix={arXiv},
}
```

## About relational contrastive (ReCo) learning
### Similarity information in ImageNet1k
``./ReCo/ImageNet1K.visual.3_hump.relation.depth_version.json`` provides the similarity information of classes in ImageNet1k (Equation 4 in the paper).

### ReCo loss
We can use ReCo loss ``./ReCo/losses.py`` in any supervised contrastive learning framework. Here we use [Parametric-Contrastive-Learning](https://github.com/dvlab-research/Parametric-Contrastive-Learning) (PaCo) in our experiments. 
```
#Run ReCo
sh ./sh/train_resnet50_reco_imagenet1k.sh
```



## Acknowledgement

Thanks to Siyu Chen (https://github.com/Siyu-C) for implementing the linear_probe. \
Thanks to Qinghong Sun for coordinating the data collection. \
Part of the ``ReCo`` code is borrowed from [Parametric-Contrastive-Learning](https://github.com/dvlab-research/Parametric-Contrastive-Learning). 

<div align="center">

![visitors](https://visitor-badge.glitch.me/badge?page_id=zhangyuanhan-ai.OmniBenchmark&left_color=green&right_color=red)

</div>



