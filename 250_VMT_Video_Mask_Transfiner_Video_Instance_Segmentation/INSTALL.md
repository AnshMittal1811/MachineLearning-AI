### Installation

First, clone the repository locally:

```bash
conda create -n vmt python=3.7 -y

conda activate vmt

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch

git clone --recursive https://github.com/SysCV/vmt.git
```

Install detectron2 for visualization under your working directory:
```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

Install dependencies and pycocotools for VIS and HQ-YTVIS:
```bash
pip install -r requirements.txt

cd cocoapi_hq/PythonAPI
# To compile and install locally 
python setup.py build_ext --inplace
# To install library to Python site-packages 
python setup.py build_ext install
```

Compiling CUDA operators:

```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py

cd ./models_swin/ops
sh ./make.sh
```

### Data Preparation

Download and extract 2019 version of YoutubeVIS train and val images with annotations from [YouTubeVIS](https://youtube-vos.org/dataset/vis/), and download [HQ-YTVIS annotations](https://www.vis.xyz/data/hqvis/) and COCO 2017 datasets. We expect the directory structure to be the following:


```
vmt
├── datasets
│   ├── coco_keepfor_ytvis19_new.json
...
ytvis
├── train
├── val
├── annotations
│   ├── instances_train_sub.json
│   ├── instances_val_sub.json
│   ├── ytvis_hq-train.json
│   ├── ytvis_hq-val.json
│   ├── ytvis_hq-test.json
coco
├── train2017
├── val2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
```

The modified coco annotations 'coco_keepfor_ytvis19_new.json' for joint training can be downloaded from [[google]](https://drive.google.com/file/d/18yKpc8wt7xJK26QFpR5Xa0vjM5HN6ieg/view?usp=sharing). The HQ-YTVIS annotations can be downloaded from [[google]](https://drive.google.com/drive/folders/1ZU8_qO8HnJ_-vvxIAn8-_kJ4xtOdkefh?usp=sharing).

##  

