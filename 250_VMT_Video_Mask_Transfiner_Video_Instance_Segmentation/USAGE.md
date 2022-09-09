### Pretrained Models

Download the pretrained models from the Model zoo table: 
```
  mkdir pretrained_model
  #And put the downloaded pretrained models in this directory.
```

### Inference & Evaluation on HQ-YTVIS

Refer to our [scripts folder](./scripts) for more commands:

Evaluating on HQ-YTVIS test:
```
bash scripts/eval_swin_test.sh
```
or 
```
bash scripts/eval_r101_test.sh
```

### Results Visualization

```
bash scripts/eval_swin_val_vis.sh
```
or
```
python3 -m tools.inference_swin_with_vis  --masks --backbone swin_l_p4w12 --output vis_output_swin_vmt --model_path ./pretrained_model/checkpoint_swinl_final.pth --save_path exp_swin_hq_val_result.json --save-frames True
```

