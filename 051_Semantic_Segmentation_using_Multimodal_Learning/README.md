# Diverse and Specific Image Captioning

This repository contains the code for __*Generating Diverse and Meaningful Captions: Unsupervised Specificity Optimization for Image Captioning (Lindh et al., 2018)*__ to appear in *Artificial Neural Networks and Machine Learning - ICANN 2018*.

A detailed description of the work, including test results, can be found in our paper: [[publisher version]](https://doi.org/10.1007/978-3-030-01418-6_18) [[author version]](https://arrow.dit.ie/airccon/13/)

Please consider citing if you use the code:
 	
	@inproceedings{lindh_generating_2018,
	series = {Lecture {Notes} in {Computer} {Science}},
	title = {Generating {Diverse} and {Meaningful} {Captions}},
	isbn = {978-3-030-01418-6},
	doi = {10.1007/978-3-030-01418-6_18},
	language = {en},
	booktitle = {Artificial {Neural} {Networks} and {Machine} {Learning} – {ICANN} 2018},
	publisher = {Springer International Publishing},
	author = {Lindh, Annika and Ross, Robert J. and Mahalunkar, Abhijit and Salton, Giancarlo and Kelleher, John D.},
	editor = {Kůrková, Věra and Manolopoulos, Yannis and Hammer, Barbara and Iliadis, Lazaros and Maglogiannis, Ilias},
	year = {2018},
	keywords = {Computer Vision, Contrastive Learning, Deep Learning, Diversity, Image Captioning, Image Retrieval, Machine Learning, MS COCO, Multimodal Training, Natural Language Generation, Natural Language Processing, Neural Networks, Specificity},
	pages = {176--187}
	}

The code in this repository builds on the code from the following two repositories:
https://github.com/ruotianluo/ImageCaptioning.pytorch  
https://github.com/facebookresearch/SentEval/  
A note is included at the top of each file that has been changed from its original state. We make these changes (and our own original files) available under Attribution-NonCommercial 4.0 International where applicable (see LICENSE.txt in the root of this repository).  
The code from the two repos listed above retain their original licenses. Please see visit their repositories for further details. The SentEval folder in our repo contains the LICENSE document for SentEval at the time of our fork.  


## Requirements  
Python 2.7 (built with the tk-dev package installed)  
PyTorch 0.3.1 and torchvision  
h5py 2.7.1  
sklearn 0.19.1  
scipy 1.0.1  
scikit-image (skimage) 0.13.1  
ijson  
Tensorflow is needed if you want to generate learning curve graphs (recommended!)  


## Setup for the Image Captioning side  
For ImageCaptioning.pytorch (previously known as neuraltalk2.pytorch) you need the pretrained resnet model found [here](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), which should be placed under `combined_model/neuraltalk2_pytorch/data/imagenet_weights`.  
You will also need the cocotalk_label.h5 and cocotalk.json from [here](https://drive.google.com/open?id=0B7fNdx_jAqhtcXp0aFlWSnJmb0k) and the pretrained Image Captioning model from the topdown directory.  
To run the prepro scripts for the Image Captioning model, first download the coco images from [link](http://mscoco.org/dataset/#download). You should put the `train2014/` and `val2014/` in the same directory, denoted as `$IMAGE_ROOT` during preprocessing.  

There’s some problems with the official COCO images. See [this issue](https://github.com/karpathy/neuraltalk2/issues/4) about manually replacing one image in the dataset. You should also run the script under utilities/check_file_types.py that will help you find one or two PNG images that are incorrectly marked as JPG images. I had to manually convert these to JPG files and replace them.  

Next, download the preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and the train-val-test splits.  
Once we have these, we can now invoke the `prepro_*.py` script, which will read all of this in and create a dataset (two feature folders, a hdf5 label file and a json file):  
```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
$ python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```
See https://github.com/ruotianluo/ImageCaptioning.pytorch for more info on the scripts if needed.  

## Setup for the Image Retrieval side  
You will need to train a SentEval model according to the instructions [here](https://github.com/facebookresearch/SentEval) using their pretrained InferSent embedder. IMPORTANT: Because of a change in SentEval, you will need to pull commit c7c7b3a instead of the latest version.  
You also need the GloVe embeddings you used for this when you’re training the full combined model.  

## Setup for the combined model  
You will need the official coco-caption evaluation code which you can find here:  
https://github.com/tylin/coco-caption  
This should go in a folder called coco_caption under   src/combined_model/neuraltalk2_pytorch  

### Run the training  
```bash
$ cd src/combined_model
$ python SentEval/examples/launch_training.py --id <your_model_id> --checkpoint_path <path_to_save_model> --start_from <directory_pretrained_captioning_model> --learning_rate 0.0000001 --max_epochs 10 --best_model_condition mean_rank --loss_function pairwise_cosine --losses_log_every 10000 --save_checkpoint_every 10000 --batch_size 2 --caption_model topdown --input_json neuraltalk2_pytorch/data/cocotalk.json --input_fc_dir neuraltalk2_pytorch/data/cocotalk_fc --input_att_dir neuraltalk2_pytorch/data/cocotalk_att --input_label_h5 neuraltalk2_pytorch/data/cocotalk_label.h5 --learning_rate_decay_start 0 --senteval_model <your_trained_senteval_model> --language_eval 1 --split val
```

The --loss_function options used for the models in the paper:  
Cos = cosine_similarity  
DP = direct_similarity  
CCos =  pairwise_cosine  
CDP = pairwise_similarity  

See combined_model/neuraltalk2_pytorch/opts.py for a list of the available parameters.  

### Run the test
```bash
$ cd src/combined_model
$ python SentEval/examples/launch_test.py --id <your_model_id> --checkpoint_path <path_to_model> --start_from <path_to_model> --load_best_model 1 --loss_function pairwise_cosine  --batch_size 2 --caption_model topdown --input_json neuraltalk2_pytorch/data/cocotalk.json --input_fc_dir neuraltalk2_pytorch/data/cocotalk_fc --input_att_dir neuraltalk2_pytorch/data/cocotalk_att --input_label_h5 neuraltalk2_pytorch/data/cocotalk_label.h5 --learning_rate_decay_start 0 --senteval_model <your_trained_senteval_model> --language_eval 1 --split test
```

To test the baseline or the latest version of a model (instead of the one marked with 'best' in the name) use:  
--load_best_model 0  
The --loss_function option will only decide which internal loss function to report the result for. No extra training will be carried out, and the other results won't be affected by this choice.  
