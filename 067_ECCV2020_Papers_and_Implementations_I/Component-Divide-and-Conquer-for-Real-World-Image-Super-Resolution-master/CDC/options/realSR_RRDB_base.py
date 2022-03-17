# coding: utf-8
import os

'''
# Train Data Root
Flat New: '/mnt/lustre/luhannan/Data_t1/photo_process/5_camera_4_warp3/train_patch/Flat_train'
PairSR_3: '/mnt/lustre/luhannan/Data_t1/photo_process/5_camera_4_warp3/train_patch/train_patch_done/'

# Test Data Root
test_1_36: '/mnt/lustre/luhannan/Data_t1/photo_process/5_camera_4_warp3/train_patch/train_patch_done/RealSR_Validation_p1'
home/weipx/RealSR/Dataset/Photo-2/train_patch
'''

import argparse

def parse_config(local_test=True):
    parser = argparse.ArgumentParser()

    # Data Preparation
#     parser.add_argument('--dataroot', type=str, default='/mnt/lustre/luhannan/Data_t1/photo_process/5_camera_4_warp3/'
#                                                         'train_patch/train_patch_done', help='Train dataset')
#     parser.add_argument('--dataroot', type=str, default='/home/weipx/RealSR/Dataset/Photo-2/train_patch', help='Train dataset')
#     parser.add_argument('--test_dataroot', type=str, default='/home/weipx/RealSR/Dataset/Photo-2/train_patch/Vali100_rand36')
    parser.add_argument('--dataroot', type=str, default='/home/xiezw/Dataset/RealSR_ICCV19/train/x3', help='Train dataset')
    parser.add_argument('--test_dataroot', type=str, default='/home/xiezw/Dataset/RealSR_ICCV19/valid/x3')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--size', type=int, default=48, help='Size of low resolution image')
    parser.add_argument('--bic', type=bool, default=False, help='')
    parser.add_argument('--rgb_range', type=float, default=1., help='255 EDSR and RCAN, 1 for the rest')
    parser.add_argument('--no_HR', type=bool, default=False, help='Whether these are HR images in testset or not')

    ## Train Settings
    parser.add_argument('--exp_name', type=str, default='RRDB-X3-RealSR', help='')
    parser.add_argument('--generatorLR', type=float, default=2e-4, help='learning rate for SR generator')
    parser.add_argument('--decay_step', type=list, default=[2e5, 4e5, 6e5, 8e5], help='')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--test_interval', type=int, default=200, help="Test epoch")
    # parser.add_argument('--save_interval', type=int, default=1, help="Test epoch")
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--workers', type=int, default=8, help='number of threads to prepare data.')

    ## SRModel Settings
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--model', type=str, default='RRDB', help='[RRDB | srresnet | EDSR | RCAN | RRDB-Fea | HGSR-MHR | HGSR]')
    parser.add_argument('--scala', type=int, default=3, help='[1 | 2 | 4], 1 for NTIRE Challenge')

    ## RRDB Settings
    parser.add_argument('--in_ch', type=int, default=3, help='Image channel, 3 for RGB')
    parser.add_argument('--out_ch', type=int, default=3, help='Image channel, 3 for RGB')
    parser.add_argument('--rrdb_nf', type=int, default=64, help='For RRDB, Feature Number for Conv')
    parser.add_argument('--rrdb_nb', type=int, default=23, help='For RRDB, Blocks Number for RRD-block')
    parser.add_argument('--rrdb_gc', type=int, default=32, help='For RRDB, Blocks Number for RRD-block')
    parser.add_argument('--rrdb_group', type=int, default=1, help='For RRDB, Blocks Number for RRD-block')

    # Content Loss Settings
    parser.add_argument('--sr_lambda', type=float, default=1, help='content loss lambda')
    parser.add_argument('--loss', type=str, default='l1', help="content loss ['l1', 'l2', 'c']")

    ## Prior Loss Settings
    parser.add_argument('--prior_loss', type=bool, default=False, help='')
    parser.add_argument('--prior_lambda', type=float, default=1, help='')
    parser.add_argument('--downsample_kernel', type=str, default='lanczos2', help='')

    ## Posterior Probability Loss
    parser.add_argument('--post_loss', type=bool, default=False, help='')
    parser.add_argument('--refine_hr_fea', type=bool, default=True, help='')
    parser.add_argument('--post_lambda', type=float, default=1, help='')

    # HGSR Settings
    parser.add_argument('--n_HG', type=int, default=4, help='number of feature maps')
    parser.add_argument('--res_type', type=str, default='res', help='residual scaling')
    parser.add_argument('--inter_supervis', type=bool, default=False, help='residual scaling')
    parser.add_argument('--inte_loss_weight', type=list, default=[1, 1, 1, 1], help='residual scaling')

    ## Contextual Loss Settings
    parser.add_argument('--cx_loss', type=bool, default=False, help='')
    parser.add_argument('--cx_loss_lambda', type=float, default=1e-1, help='Weight for CX_Loss')
    parser.add_argument('--cx_vgg_layer', type=int, default=34, help='[17 | 34]')

    ## VGG Loss Settings
    parser.add_argument('--vgg_loss', type=bool, default=False, help='Whether downsample test LR image')
    parser.add_argument('--vgg_lambda', type=float, default=1, help='learning rate for generator')
    parser.add_argument('--vgg_loss_type', type=str, default='l1', help="loss L1 or L2 ['l1', 'l2']")
    parser.add_argument('--vgg_layer', type=str, default=34, help='[34 | 35]')

    ## TV Loss Settings
    parser.add_argument('--tv_loss', type=bool, default=False, help='Whether use tv loss')
    parser.add_argument('--tv_lambda', type=float, default=0.1, help='tv loss lambda')

    ## Noise Settings
    parser.add_argument('--add_noise', type=bool, default=False, help='Whether downsample test LR image')
    parser.add_argument('--noise_level', type=float, default=0.1, help='learning rate for SR generator')

    ## Mix Settings
    parser.add_argument('--mix_bic_real', type=bool, default=False, help='Whether mix ')

    ## Default Settings
    parser.add_argument('--gpus', type=int, default=1, help='Placeholder, will be changed in run_train.sh')
    parser.add_argument('--train_file', type=str, default='', help='placeholder, will be changed in main.py')
    parser.add_argument('--config_file', type=str, default='', help='placeholder, will be changed in main.py')

    opt = parser.parse_args()

    return opt

























