CUDA_VISIBLE_DEVICES=0 python3 -m tools.inference  --masks --backbone resnet101 --model_path ./pretrained_model/checkpoint_r101_final.pth --save_path exp_r101_hq_val_result.json 
