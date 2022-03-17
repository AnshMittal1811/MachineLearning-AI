# coding: utf-8
import os
"""
Config File for HGSR-Multi-Scale-Supervision
"""

import argparse

def parse_config(local_test=True):
    parser = argparse.ArgumentParser()

    # Data Preparation
    parser.add_argument('--dataroot', type=str, default='/media/data1/xzw/Datasets/Photo-4/train_patch')
    parser.add_argument('--test_dataroot', type=str, default='/media/data1/xzw/Datasets/Photo-4/Vali100_rand36')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--size', type=int, default=48, help='Size of low resolution image')
    parser.add_argument('--bic', type=bool, default=False, help='')
    parser.add_argument('--rgb_range', type=float, default=1., help='255 EDSR and RCAN, 1 for the rest')
    parser.add_argument('--no_HR', type=bool, default=False, help='Whether these are HR images in testset or not')

    ## Train Settings
    parser.add_argument('--exp_name', type=str, default='CDC-X4', help='')
    parser.add_argument('--generatorLR', type=float, default=2e-4, help='learning rate for SR generator')
    parser.add_argument('--decay_step', type=list, default=[2e5, 4e5, 6e5, 8e5], help='')
#     parser.add_argument('--generatorLR', type=float, default=1e-4, help='learning rate for SR generator')
#     parser.add_argument('--decay_step', type=list, default=[5e4, 1e5, 2e5, 3e5], help='')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--test_interval', type=int, default=1, help="Test epoch")
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--workers', type=int, default=8, help='number of threads to prepare data.')

    ## SRModel Settings
    parser.add_argument('--pretrain', type=str, default='')
#     parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--model', type=str, default='HGSR-MHR', help='[RRDB_net | srresnet | EDSR | RCAN | HGSR | HGSR-MHR]')
    parser.add_argument('--scala', type=int, default=4, help='[1 | 2 | 4], 1 for NTIRE Challenge')
    parser.add_argument('--in_ch', type=int, default=3, help='Image channel, 3 for RGB')
    parser.add_argument('--out_ch', type=int, default=3, help='Image channel, 3 for RGB')

    # HGSR Settings
    parser.add_argument('--n_HG', type=int, default=6, help='number of feature maps')
    parser.add_argument('--res_type', type=str, default='res', help='residual scaling')
    parser.add_argument('--inter_supervis', type=bool, default=True, help='residual scaling')
    parser.add_argument('--mscale_inter_super', type=bool, default=False, help='residual scaling')
#     parser.add_argument('--inte_loss_weight', type=list, default=[2e-1, 4e-1, 6e-1, 1], help='residual scaling')
    parser.add_argument('--inte_loss_weight', type=list, default=[1, 2, 5, 1], help='residual scaling')

    # Content Loss Settings
    parser.add_argument('--sr_lambda', type=float, default=1, help='content loss lambda')
    parser.add_argument('--loss', type=str, default='l1', help="content loss ['l1', 'l2', 'c', 'sobel']")

    ## Prior Loss Settings
    parser.add_argument('--prior_loss', type=bool, default=False, help='')
    parser.add_argument('--prior_lambda', type=float, default=1, help='')
    parser.add_argument('--downsample_kernel', type=str, default='lanczos2', help='')

    ## Contextual Loss Settings
    parser.add_argument('--cx_loss', type=bool, default=False, help='')
    parser.add_argument('--cx_loss_lambda', type=float, default=1e-1, help='Weight for CX_Loss')
    parser.add_argument('--cx_vgg_layer', type=int, default=34, help='[17 | 34]')

    ## VGG Loss Settings
    parser.add_argument('--vgg_loss', type=bool, default=False, help='Whether downsample test LR image')
    parser.add_argument('--vgg_lambda', type=float, default=1, help='learning rate for generator')
    parser.add_argument('--vgg_loss_type', type=str, default='l1', help="loss L1 or L2 ['l1', 'l2']")
#     parser.add_argument('--vgg_layer', type=str, default=34, help='[34 | 35]')
#     parser.add_argument('--vgg', type=str, default='./models/vgg19-dcbb9e9d.pth', help='number of threads to prepare data.')

    ## TV Loss Settings
    parser.add_argument('--tv_loss', type=bool, default=False, help='Whether use tv loss')
    parser.add_argument('--tv_lambda', type=float, default=0.1, help='tv loss lambda')
    
    ## Entropy Loss Settings
    parser.add_argument('--entropy_loss', type=bool, default=False, help='Whether use entropy loss')
    parser.add_argument('--entropy_lambda', type=float, default=0.5, help='entropy loss lambda')
    
    ## Sobel Loss Settings
    parser.add_argument('--sobel_loss', type=bool, default=False, help='Whether use sobel loss')
    parser.add_argument('--sobel_lambda', type=float, default=1, help='sobel loss lambda')
    
    ## Harris Loss Settings
    parser.add_argument('--harris_loss', type=bool, default=False, help='Whether use harris loss')
    parser.add_argument('--corner_lambda', type=float, default=5, help='corner loss lambda')
    parser.add_argument('--edge_lambda', type=float, default=2, help='edge loss lambda')
    
    ## VGG Loss Settings
    parser.add_argument('--correctness', type=bool, default=False, help='Whether downsample test LR image')
    parser.add_argument('--flow_lambda', type=float, default=0.1, help='learning rate for generator')
    parser.add_argument('--vgg_layer', type=str, default=9, help='[9 | 36]')
    parser.add_argument('--vgg', type=str, default='./models/vgg19-dcbb9e9d.pth', help='number of threads to prepare data.')

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

























