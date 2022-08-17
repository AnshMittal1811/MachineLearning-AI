export DEBUG=""
export USE_SLURM=0

### Train FWD-D model
python train.py --num_workers 16 \
    --accumulation 'alphacomposite' --num_views 4 --resume \
    --dataset_path "/data/path/dtu_down_4" \
    --dataset 'dtu' --log-dir '/your/project/checkpoint/path/log/%s/' \
    --model_type 'multi_z_transformer' --max_epoch 20000  \
    --norm_G 'sync:spectral_batch'  --render_ids 1 \
    --suffix '' --lr 0.0001 --lr_g 0.0001 --beta2 0.999 \
    --atten_n_head 4 --use_transformer --atten_k_dim 16 --atten_v_dim 64\
    --losses '5.0_l2' '1.0_content' --pp_pixel 24 --H 300 --W 400\
    --batch-size 16 --folder 'multi_synsin_temp' --gpu_ids 0,1,2,3 \
    --refine_model_type 'resnet_256W8customup' --radius 1.5 --decoder_norm "instance"\
    --debug_path "/your/project/checkpoint/path" \
	--normalize_image --normalize_depth --input_view_num 3 --inverse_depth_com --use_tanh --view_dependence --down_sample --append_RGB --use_gt_depth --depth_com --train_depth --gt_depth_loss_weight 3.0\
    --depth_regressor "unet" --depth_lr_scaling 1.0

### Train FWD model
# python train.py --num_workers 16 \
#     --accumulation 'alphacomposite' --num_views 4 --resume \
#     --dataset_path "/data/path/dtu_down_4" \
#     --dataset 'dtu' --log-dir '/your/project/checkpoint/path/log/%s/' \
#     --model_type 'multi_z_transformer' --max_epoch 20000  \
#     --norm_G 'sync:spectral_batch'  --render_ids 1 \
#     --suffix '' --lr 0.0001 --lr_g 0.0001 --beta2 0.999 \
#     --atten_n_head 4 --use_transformer --atten_k_dim 16 --atten_v_dim 64\
#     --losses '5.0_l2' '1.0_content' --pp_pixel 16 --H 300 --W 400\
#     --batch-size 16 --folder 'multi_synsin_temp' --gpu_ids 0,1,2,3 \
#     --refine_model_type 'resnet_256W8customup' --radius 1.5 \
#     --debug_path "/your/project/checkpoint/path"\
# 	  --normalize_image --normalize_depth --input_view_num 3 --inverse_depth_com --use_tanh --view_dependence --down_sample --append_RGB --use_gt_depth --depth_com\
#     --depth_regressor "unet" --depth_lr_scaling 1.0 --learnable_mvs --mvs_depth --pretrained_MVS