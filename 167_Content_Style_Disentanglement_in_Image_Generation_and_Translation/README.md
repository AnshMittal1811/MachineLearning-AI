# DiagonalGAN
## Official Pytorch Implementation of "Diagonal Attention and Style-based GAN for Content-Style Disentanglement in Image Generation and Translation" (ICCV 2021)
### [Gihyun Kwon](https://sites.google.com/view/gihyunkwon), [Jong Chul Ye](https://bispl.weebly.com/professor.html)

Arxiv : [link](https://arxiv.org/abs/2103.16146)
 CVF : [link](https://openaccess.thecvf.com/content/ICCV2021/papers/Kwon_Diagonal_Attention_and_Style-Based_GAN_for_Content-Style_Disentanglement_in_Image_ICCV_2021_paper.pdf)

### Contact
If you have any question, 

e-mail : cyclomon@kaist.ac.kr

### Abstract
One of the important research topics in image generative models is to disentangle the spatial contents and styles for their separate control. Although StyleGAN can generate content feature vectors from random noises, the resulting spatial content control is primarily intended for minor spatial variations, and the disentanglement of global content and styles is by no means complete. Inspired by a mathematical understanding of normalization and attention, here we present a novel hierarchical adaptive Diagonal spatial ATtention (DAT) layers to separately manipulate the spatial contents from styles in a hierarchical manner. Using DAT and AdaIN, our method enables coarse-to-fine level disentanglement of spatial contents and styles. In addition, our generator can be easily integrated into the GAN inversion framework so that the content and style of translated images from multi-domain image translation tasks can be flexibly controlled. By using various datasets, we confirm that the proposed method not only outperforms the existing models in disentanglement scores, but also provides more flexible control over spatial features in the generated images.


![Models9](https://user-images.githubusercontent.com/88644048/130436052-f9c213b3-a3f4-403f-84b9-9ccdad8c8970.png)

### Citation
```
@inproceedings{kwon2021diagonal,
  title={Diagonal attention and style-based GAN for content-style disentanglement in image generation and translation},
  author={Kwon, Gihyun and Ye, Jong Chul},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13980--13989},
  year={2021}
}
```

### Environment Settings
Python 3.6.7 +

Pytorch 1.5.0 +

### Dataset
For faster training, we recommend .jpg file format.

Download Link:
[CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ) / 
[AFHQ](https://github.com/clovaai/stargan-v2)

Unzip the files and put the folder into the data directory (```./data/Celeb/data1024``` , ```./data/afhq```)

To process the data for multidomain Diagonal GAN, run 

```
./data/Celeb/Celeb_proc.py 
```
After download the CelebA-HQ dataset to save males / females images in different folders.

We randomly selected 1000 images as validation set for each domain (1000 males / 1000 females).

Save validation files into ```./data/Celeb/val/males``` , ```./data/Celeb/val/females```


## Train
### Train Basic Diagonal GAN
For full-resolution CelebA-HQ training,

```
python train.py --datapath ./data/Celeb/data1024 --sched --max_size 1024 --loss r1
```

For full-resolution AFHQ training,

```
python train.py --datapath ./data/afhq --sched --max_size 512 --loss r1
```
### Train Multidomain Diagonal GAN
For training multidomain (Males/ Females) models, run

```
python train_multidomain.py --datapath ./data/Celeb/mult --sched --max_size 256
```

### Train IDInvert Encoders on pre-trained Multidomain Diagonal GAN
For training IDInvert on  pre-trained model,
```
python train_idinvert.py --ckpt $MODEL_PATH$ 
```

or you can download the pre-trained Multidomain model. 

Save the model in ```./checkpoint/train_mult/CelebAHQ_mult.model```

and set ```$MODEL_PATH$``` as above.

### Additional latent code optimization ( for inference )
To further optimize the latent codes, 

```
python train_idinvert_opt.py --ckpt $MODEL_PATH$ --enc_ckpt $ENC_MODEL_PATH$
```

```MODEL_PATH``` is pre-trained multidomain model directory, and

```ENC_MODEL_PATH``` is IDInvert encoder model directory.

You can download the pre-trained IDInvert encoder models.

We also provide optimized latent codes. 

## Pre-trained model Download


Pre-trained Diagonal GAN on 1024x1024 CelebA-HQ : [Link](https://drive.google.com/drive/folders/1VvLNwNIaquXz9tKZKXI4xMwCzCt6tN3k?usp=sharing)
save to ```./checkpoint/train_basic```

Pre-trained Diagonal GAN on 512x512 AFHQ : [Link](https://drive.google.com/drive/folders/1VvLNwNIaquXz9tKZKXI4xMwCzCt6tN3k?usp=sharing)
save to ```./checkpoint/train_basic```

Pre-trained Multidomain Diagonal GAN on 256x256 CelebA-HQ : [Link](https://drive.google.com/drive/folders/1R00015UnqQk6KZZugvwy79xoZsdAOPuF?usp=sharing)
save to ```./checkpoint/train_mult```

Pre-trained IDInvert Encoders on 256x256 CelebA-HQ : [Link](https://drive.google.com/drive/folders/1or9QzF5wiO4LUczpAPV3tLNquHJgG865?usp=sharing)
save to ```./checkpoint/train_idinvert```

Optimized latent codes : [Link](https://drive.google.com/drive/folders/1DBLLm45tdjMMD42Xp_JI1m1U-yHF53rf?usp=sharing)
save to ```./codes```

## Generate Images
To generate the images from the pre-trained model,

```
python generate.py --mode $MODE$ --domain $DOM$ --target_layer $TARGET$
```

for ```$MODE$```, there is three choices  (sample , mixing, interpolation).

using 'sample' just sample random samples, 

for 'mixing', generate images with random code on target layer ```$TARGET$```

for 'interpolate', generate with random interpolation on target layer ```$TARGET$```

also, we can choose style or content with setting ```$DOM$``` with 'style' or 'content'


## Generate Images on Inverted model
To generate the images from the pre-trained IDInvert,

```
python generate_idinvert.py --mode $MODE$ --domain $DOM$ --target_layer $TARGET$
```

for ```$MODE$```, there is three choices  (sample , mixing, encode).

using 'sample' just sample random samples, 

for 'mixing', generate images with random code on target layer ```$TARGET$```

for 'encode', generate auto-encoder reconstructions

we can choose style or content with setting ```$DOM$``` with 'style' or 'content'

To use additional optimized latent codes, activate ```--use_code```


## Examples

```
python generate.py --mode sample 
```
![03_content_sample](https://user-images.githubusercontent.com/88644048/135963795-e2196dfe-b55e-431e-b119-550d5a0be9ce.jpg)

8x8 resolution content
```
python generate.py --mode mixing --domain content --target_layer 2 3
```
![03_content_mixing](https://user-images.githubusercontent.com/88644048/135963856-56d9d83f-a72e-497d-b8f9-c6f6729d644c.jpg)


High resolution style
```
python generate.py --mode mixing --domain style --target_layer 14 15 16 17
```
![02_style_mixing](https://user-images.githubusercontent.com/88644048/135963909-f0fa988a-c8f5-4920-ba07-e8719d0a69d5.jpg)

