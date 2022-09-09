## Visdom
Monitoring of the training/evaluation progress is possible via command line as well as [Visdom](https://github.com/fossasia/visdom). 
For the latter, a Visdom server must be running at `VISDOM_PORT=8090` and `VISDOM_SERVER=http://localhost`. 
To deactivate Visdom logging set `VISDOM_ON=False`.

## Train
We provide configurations files under `configs/deformable_mask_head/` and `configs/devis/` to train the Mask-Head  and DeVIS respectively.
In order to launch a training you just need to simply specify the number of GPUS using `--nproc_per_node` and the corresponding config file after `--config-file`. 
For instance, the command for training YT-VIS 2019 model with 4GPUs is as following:

```
torchrun --nproc_per_node=4 main.py --config-file configs/devis/YT-19/devis_R_50_YT-19.yaml
```
User can also override config file parameters by passing the new KEY VALUE pair. 
For instance, to double the default lr:

```
torchrun --nproc_per_node=4 main.py --config-file configs/devis/YT-19/devis_R_50_YT-19.yaml SOLVER.BASE_LR 0.0002
```

## Model zoo

| Dataset          | Backbone | AP   | Total <br/>batch size | Training<br/> GPU hours \* | Max GPU <br/>memory | URL                                                                                                                                                                                                                                                                                             |
|------------------|-------|------|-----------------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| COCO             | R50   | 38.0 | 14                    | 345                        | 27GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/deformable_mask_head/deformable_mask_head_R_50.yaml) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/coco/r50_deformable_detr_segmentation.zip)                                                                 |
| COCO             | R101  | 39.9 | 14                    | 260                        | 32GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/deformable_mask_head/deformable_mask_head_R_101.yaml) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/coco/r101_deformable_detr_segmentation.zip)                                                               |
| COCO             | SwinL | 45.2 | 7                     | 470                        | 26GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/deformable_mask_head/deformable_mask_head_SwinL.yaml) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/coco/swinL_deformable_detr_segmentation.zip)                                                              |
| YouTube-VIS 19   | R50   | 44.4 | 4                     | 120                        | 18GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/YT-19/devis_R_50_YT-19.yaml) <br/>[log](https://vision.in.tum.de/webshare/u/cad/model_zoo/yt-19/log_r50_devis_yt-19.txt) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/yt-19/r50_devis_yt-19.zip)       |
| YouTube-VIS 19   | SwinL | 57.1 | 4                     | 220                        | 37GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/YT-19/devis_Swin_L_YT-19.yaml) <br/>[log](https://vision.in.tum.de/webshare/u/cad/model_zoo/yt-19/log_swinL_devis_yt-19.txt) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/yt-19/swinL_devis_yt-19.zip) |
| YouTube-VIS 21   | R50   | 43.1 | 4                     | 200                        | 24GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/YT-21/devis_R_50_YT-21.yaml) <br/>[log](https://vision.in.tum.de/webshare/u/cad/model_zoo/yt-21/log_r50_devis_yt-21.txt) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/yt-21/r50_devis_yt-21.zip)       |
| YouTube-VIS 21** | SwinL | 54.4 | 4                     | 305                        | 40GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/YT-21/devis_Swin_L_YT-21.yaml) <br/>[log](https://vision.in.tum.de/webshare/u/cad/model_zoo/yt-21/log_swinL_devis_yt-21.txt) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/yt-21/swinL_devis_yt-21.zip) |
| OVIS             | R50   | 23.7 | 4                     | 145                        | 24GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/OVIS/devis_R_50_OVIS.yaml) <br/>[log](https://vision.in.tum.de/webshare/u/cad/model_zoo/ovis/log_r50_devis_ovis.txt) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/ovis/r50_devis_ovis.zip)             |
| OVIS             | SwinL | 35.5 | 4                     | 204                        | 38GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/OVIS/devis_Swin_L_OVIS.yaml) <br/>[log](https://vision.in.tum.de/webshare/u/cad/model_zoo/ovis/log_swinL_devis_ovis.txt) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/ovis/swinL_devis_ovis.zip)       |

** We have used the following [train set](https://vision.in.tum.de/webshare/u/cad/instances_wo_2975_2359.zip) in order to fit GPU memory, which removes the 2 most crowded train videos.  
## Ablations
We also provide configuration file to run all the ablation studies presented on Table 1:

| Method                       | Clip size | K_temp | Features<br/> scales | AP   | Training<br/> GPU hours\* | Max GPU <br/>memory | URL                                                                                                                        |
|------------------------------|----------|--------|----------------------|------|---------------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------|
| Deformable VisTR             | 36       | 4      | 1                    | 34.2 | 190                       | 10GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/ablations/devis_ablation0_deformable_vistr.yaml)    |
| Deformable VisTR             | 36       | 0      | 1                    | 35.3 | 150                       | 7GB                 | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/ablations/devis_ablation1_deformable_vistr_wo_temp_conn.yaml) |
| Deformable VisTR             | 6        | 0      | 1                    | 32.4 | 40                        | 2GB                 | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/ablations/devis_ablation2-5_single-scale_wo_temp_conn.yaml)   |
| DeVIS                        | 6        | 4      | 1                    | 34.0 | 46                        | 3GB                 | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/ablations/devis_ablation2_single-scale.yaml)                  |
| +increase spatial inputs     | 6        | 4      | 4                    | 35.9 | 104                       | 15GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/ablations/devis_ablation3_increased-spatial-inputs.yaml)      |
| +instance aware obj. queries | 6        | 4      | 4                    | 37.0 | 115                       | 15GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/ablations/devis_ablation4_instance-aware.yaml)                |
| +multi-scale mask head       | 6        | 4      | 4                    | 40.2 | 128                       | 16GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/ablations/devis_ablation5_multi-scale_mask-head.yaml)         |
| +multi-cue clip tracking     | 6        | 4      | 4                    | 41.9 | ---                       | --                  | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/ablations/devis_ablation6_TEST_multi-cue_tracking.yaml)       |
| +aux. loss weighting         | 6        | 4      | 4                    | 44.0 | 128                       | 16GB                | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/YT-19/devis_R_50_YT-19.yaml)                              |


*Training GPU hours measured on a RTX A6000 GPU

## Validation during training
We support evaluation during training for VIS datasets despite GT annotations not available.
Results will be saved into `TEST.SAVE_PATH` folder, created inside `OUTPUT_DIR`.
Users can set `EVAL_PERIOD` to select the interval between validations (0 to disable it) 
Additionally, `START_EVAL_EPOCH` allows selecting at which epoch start considering `EVAL_PERIOD`, useful in order to omit first epochs.