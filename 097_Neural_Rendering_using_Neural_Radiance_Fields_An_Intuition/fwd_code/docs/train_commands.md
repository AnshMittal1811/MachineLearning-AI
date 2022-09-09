# Training & Evaluation

### Training Instructions on DTU

The model on DTU dataset could be trained by the command ```bash train.sh```.
Please provide the paths of ```--dataset_path, --log-dir, --debug_path```, and choose model types by committing/un-committing codes in ```train.sh```. 

### Evaluation Instructions on DTU
The model could be evaluated by the command ```bash eval.sh```.
Our evaluation follows the protocol of [PixelNeRF](https://github.com/sxyu/pixel-nerf), where several fixed input views are given as inputs for synthesizing the rest views.

During evaluation, we calculate metrics for all the rest views, as well as metrics for selected views as PixelNeRF.
Moreover, input images and input view number could be specified by ```--src_list``` and ```input_view```.

### Pre-trained Model
The pre-trained model weight files are hosted in [this link](https://drive.google.com/drive/u/1/folders/1KQvUpX3ZI6JFPjZM5OkTf4hKTVYTgUQ8).