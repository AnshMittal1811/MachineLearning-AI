# coding: utf-8
import os
"""
SRResNet 在real pair上实验，使用真实图像对
"""

'''
# Train Data Root
’/mnt/lustre/luhannan/Data_t1/PairHL_crop800_6a/‘
'/mnt/lustre/luhannan/Data_t1/PairHL_crop800_5c/'
'/mnt/lustrenew/luhannan/dataset/PairHL_crop800_7b' # X2

# Test Data Root
'./testsets/test_c400_x2/'
'./testsets/test_HL_c200/'
'./testsets/real_vali_c200_20/'
'''

import argparse

def parse_config(local_test=True):
    parser = argparse.ArgumentParser()

    # Data Preparation
    parser.add_argument('--dataroot', type=str, default='/home/xiezw/Dataset/RealSR_ICCV19/train/x2')
    parser.add_argument('--test_dataroot', type=str, default='/home/xiezw/Dataset/RealSR_ICCV19/valid/x2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--size', type=int, default=48, help='Size of low resolution image')
    parser.add_argument('--bic', type=bool, default=False, help='')
    parser.add_argument('--rgb_range', type=float, default=255., help='255 EDSR and RCAN, 1 for the rest')
    parser.add_argument('--no_HR', type=bool, default=False, help='Whether these are HR images in testset or not')

    ## Train Settings
    parser.add_argument('--exp_name', type=str, default='RCAN-X2-RealSR', help='')
    parser.add_argument('--generatorLR', type=float, default=2e-4, help='learning rate for SR generator')
    # parser.add_argument('--decay_step', type=list, default=[2e5, 4e5, 6e5, 8e5], help='')
    parser.add_argument('--decay_step', type=list, default=[2e5, 4e5, 6e5, 8e5], help='')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--test_interval', type=int, default=200, help="Test epoch")
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--workers', type=int, default=8, help='number of threads to prepare data.')

    ## SRModel Settings
    # parser.add_argument('--pretrain', type=str, default='./models/RCAN_BIX2.pt')
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--model', type=str, default='RCAN', help='[RRDB_net | srresnet | EDSR | RCAN]')
    parser.add_argument('--sr_norm_type', default=None, help='For SRResNet, IN or No IN [None | IN ]')
    parser.add_argument('--scala', type=int, default=2, help='[1 | 2 | 4], 1 for NTIRE Challenge')
    parser.add_argument('--in_ch', type=int, default=3, help='Image channel, 3 for RGB')
    parser.add_argument('--out_ch', type=int, default=3, help='Image channel, 3 for RGB')

    # RCAN Settings
    parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
    parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
    parser.add_argument('--skip_threshold', type=float, default='10', help='skipping batch that has large error')

    # EDSR Settings
    parser.add_argument('--act', type=str, default='relu', help='activation function')
    # parser.add_argument('--n_resblocks', type=int, default=32, help='number of residual blocks')
    # parser.add_argument('--n_feats', type=int, default=256, help='number of feature maps')
    # parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
    parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
    parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
    parser.add_argument('--n_colors', type=int, default=3, help='number of feature maps')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')

    # Content Loss Settings
    parser.add_argument('--sr_lambda', type=float, default=1, help='content loss lambda')
    parser.add_argument('--loss', type=str, default='l1', help="content loss ['l1', 'l2', 'c']")

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

























