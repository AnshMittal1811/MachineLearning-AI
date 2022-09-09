CUDA_VISIBLE_DEVICES=0 python3 -m tools.inference_swin_test  --masks --backbone swin_l_p4w12 --model_path ./pretrained_model/checkpoint_swinl_final.pth --save_path exp_swin_hq_test_result.json 


