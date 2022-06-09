#!/bin/bash

mod='snp_0' # try the batch script

dataset=$1 # command line input
scene=$2 # command line input

tb_log_dir='./tb'
mkdir -p $tb_log_dir

if [ "$dataset" == 'LLFF' ]
then
   precomputed_depth_path="./data/LLFF/depths"
   max_num_pts=500000
   radius=1.0e-3
   render_scale=2
   rasterize_rounds=2
   crop_h=1512
   crop_w=2016


elif [ "$dataset" == 'DTUHR' ]
then
   precomputed_depth_path="./data/DTU/depths"
   max_num_pts=4700000
   radius=1.5e-3
   render_scale=1
   rasterize_rounds=2
   crop_h=1200
   crop_w=1600

else
   echo "unsupported dataset"
   exit 1
fi

free_xyz=0
free_opy=1
free_rad=0
gamma=1e-3
loss='l1'
batch_size=1
shader_arch='simple_unet'
shader_norm='none' # works best
feat_smooth_loss_coeff=0.01
do_random_affine=0

lr=1e-4
lr_feat=1e-2
lr_opy=1e-4

# round0 setting
# run round0 to extract error map
do_2d_shading=0
pts_dropout_rate=0.0
num_steps=5000
img_log_freq=50
VAL_FREQ=1000
dim_pointfeat=27
shader_output_channel=3
basis_type='SH'

name_base="d${dataset}_${mod}_${scene}_b${batch_size}_l${loss}_lr${lr}_lrf${lr_feat}_lro${lr_opy}_fo${free_opy}_r${radius}_g${gamma}_s2d${do_2d_shading}_pd${pts_dropout_rate}_dimf${dim_pointfeat}_so${shader_output_channel}_bs${basis_type}_sa${shader_arch}_sn${shader_norm}_fs${feat_smooth_loss_coeff}_aff${do_random_affine}"
name0="r0_${name_base}"
CUDA_VISIBLE_DEVICES=0 python train_val.py --setting "$dataset"  --crop_h 1200 --crop_w 1600 \
--resize_h 1200 --resize_w 1600 \
--name "$name0" \
--batch_size $batch_size --SUM_FREQ 100 \
--tb_log_dir $tb_log_dir \
--num_steps $num_steps --IMG_LOG_FREQ $img_log_freq --VAL_FREQ $VAL_FREQ \
--single "$scene" \
--HR 1 \
--precomputed_depth_path $precomputed_depth_path \
--loss_type $loss --feat_smooth_loss_coeff $feat_smooth_loss_coeff --do_random_affine $do_random_affine \
--free_xyz $free_xyz --free_opy $free_opy --blend_gamma $gamma --sphere_radius $radius --lr $lr \
--render_scale 1 --do_2d_shading $do_2d_shading --shader_arch $shader_arch --pts_dropout_rate $pts_dropout_rate \
--dim_pointfeat $dim_pointfeat \
--shader_output_channel $shader_output_channel --basis_type $basis_type --shader_norm $shader_norm \
--special_args_dict "vert_feat:${lr_feat},vert_opy:${lr_opy}"

# round 0.1. load round0 and add points
name0_1="r0.1_${name_base}"
CUDA_VISIBLE_DEVICES=0 python train_val.py --setting "$dataset"  --crop_h 1200 --crop_w 1600 \
--resize_h 1200 --resize_w 1600 \
--name "$name0_1" \
--batch_size $batch_size --SUM_FREQ 100 \
--tb_log_dir $tb_log_dir \
--num_steps $num_steps --IMG_LOG_FREQ $img_log_freq --VAL_FREQ $VAL_FREQ \
--single "$scene" \
--HR 1 \
--precomputed_depth_path $precomputed_depth_path \
--loss_type $loss --feat_smooth_loss_coeff $feat_smooth_loss_coeff --do_random_affine $do_random_affine \
--free_xyz $free_xyz --free_opy $free_opy --blend_gamma $gamma --sphere_radius $radius --lr $lr \
--render_scale 1 --do_2d_shading $do_2d_shading --shader_arch $shader_arch --pts_dropout_rate $pts_dropout_rate \
--dim_pointfeat $dim_pointfeat \
--shader_output_channel $shader_output_channel --basis_type $basis_type --shader_norm $shader_norm \
--special_args_dict "vert_feat:${lr_feat},vert_opy:${lr_opy}" \
--pointadd_only 1 --restore_ckpt "./checkpoints/${name0}.pth"

# round 1. do the final training with all components
do_2d_shading=1
pts_dropout_rate=0.5
num_steps=50000
img_log_freq=500
VAL_FREQ=2000
dim_pointfeat=288
shader_output_channel=32
basis_type='SH'

name_base="d${dataset}_${mod}_${scene}_b${batch_size}_l${loss}_lr${lr}_lrf${lr_feat}_lro${lr_opy}_fo${free_opy}_r${radius}_g${gamma}_s2d${do_2d_shading}_pd${pts_dropout_rate}_dimf${dim_pointfeat}_so${shader_output_channel}_bs${basis_type}_sa${shader_arch}_sn${shader_norm}_fs${feat_smooth_loss_coeff}_aff${do_random_affine}"
name1="r1_${name_base}"

CUDA_VISIBLE_DEVICES=0 python train_val.py --setting "$dataset"  --crop_h $crop_h --crop_w $crop_w \
--resize_h $crop_h --resize_w $crop_w \
--name "$name1" \
--batch_size $batch_size --SUM_FREQ 100 \
--tb_log_dir $tb_log_dir \
--num_steps $num_steps --IMG_LOG_FREQ $img_log_freq --VAL_FREQ $VAL_FREQ \
--single "$scene" \
--HR 1 \
--precomputed_depth_path $precomputed_depth_path \
--loss_type $loss --feat_smooth_loss_coeff $feat_smooth_loss_coeff --do_random_affine $do_random_affine \
--free_xyz $free_xyz --free_opy $free_opy --blend_gamma $gamma --sphere_radius $radius --lr $lr \
--render_scale $render_scale --do_2d_shading $do_2d_shading --shader_arch $shader_arch --pts_dropout_rate $pts_dropout_rate \
--dim_pointfeat $dim_pointfeat \
--shader_output_channel $shader_output_channel --basis_type $basis_type --shader_norm $shader_norm \
--special_args_dict "vert_feat:${lr_feat},vert_opy:${lr_opy}" \
--restore_pointclouds "./pointclouds/${name0_1}.pt" \
--max_num_pts $max_num_pts --rasterize_rounds $rasterize_rounds