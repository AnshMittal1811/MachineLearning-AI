"""
Rethinking Portrait Matting with Privacy Preserving
evaluation file.

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting
Paper link: https://arxiv.org/abs/2203.16828

"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from metrics import *
from util import listdir_nohidden


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the training purpose.')
    parser.add_argument('--pred_dir', type=str, required=True, help='path to predictions')
    parser.add_argument('--alpha_dir', type=str, required=True, help='path to groundtruth alpha')
    parser.add_argument('--trimap_dir', type=str, help='path to trimap')
    parser.add_argument('--fast_test', action='store_true', help='skip grad and conn')
    args, _ = parser.parse_known_args()
    
    print('Prediction dir: {}'.format(args.pred_dir))
    print('Alpha dir: {}'.format(args.alpha_dir))
    print('Trimap dir: {}'.format(args.trimap_dir))
    if args.fast_test:
        print('Will skip gradient and connectivity...')
    return args

def evaluate_folder(args):
    img_list = listdir_nohidden(args.alpha_dir)
    total_number = len(img_list)

    sad_diffs = 0.
    mse_diffs = 0.
    mad_diffs = 0.
    sad_trimap_diffs = 0.
    mse_trimap_diffs = 0.
    mad_trimap_diffs = 0.
    sad_fg_diffs = 0.
    sad_bg_diffs = 0.
    conn_diffs = 0.
    grad_diffs = 0.

    for img_name in tqdm(img_list):
        predict = cv2.imread(os.path.join(args.pred_dir, img_name), 0).astype(np.float32)/255.0
        alpha = cv2.imread(os.path.join(args.alpha_dir, img_name), 0).astype(np.float32)/255.0
        
        if args.trimap_dir is not None:
            trimap = cv2.imread(os.path.join(args.trimap_dir, img_name), 0).astype(np.float32)
        else:
            trimap = None
        
        sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(predict, alpha)

        if trimap is not None:
            sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(predict, alpha, trimap)
            sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(predict, alpha, trimap)
        else:
            sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = 0.0, 0.0, 0.0
            sad_fg_diff, sad_bg_diff = 0.0, 0.0
        
        if args.fast_test:
            conn_diff = 0.0
            grad_diff = 0.0
        else:
            conn_diff = compute_connectivity_loss_whole_image(predict, alpha)
            grad_diff = compute_gradient_whole_image(predict, alpha)
        

        sad_diffs += sad_diff
        mse_diffs += mse_diff
        mad_diffs += mad_diff
        mse_trimap_diffs += mse_trimap_diff
        sad_trimap_diffs += sad_trimap_diff
        mad_trimap_diffs += mad_trimap_diff
        sad_fg_diffs += sad_fg_diff
        sad_bg_diffs += sad_bg_diff
        conn_diffs += conn_diff
        grad_diffs += grad_diff

    res_dict = {}
    res_dict['SAD'] = sad_diffs / total_number
    res_dict['MSE'] = mse_diffs / total_number
    res_dict['MAD'] = mad_diffs / total_number
    res_dict['SAD_TRIMAP'] = sad_trimap_diffs / total_number
    res_dict['MSE_TRIMAP'] = mse_trimap_diffs / total_number
    res_dict['MAD_TRIMAP'] = mad_trimap_diffs / total_number
    res_dict['SAD_FG'] = sad_fg_diffs / total_number
    res_dict['SAD_BG'] = sad_bg_diffs / total_number
    res_dict['CONN'] = conn_diffs / total_number
    res_dict['GRAD'] = grad_diffs / total_number

    print('Average results')
    print('Test image numbers: {}'.format(total_number))
    print('Whole image SAD:', res_dict['SAD'])
    print('Whole image MSE:', res_dict['MSE'])
    print('Whole image MAD:', res_dict['MAD'])
    print('Unknown region SAD:', res_dict['SAD_TRIMAP'])
    print('Unknown region MSE:', res_dict['MSE_TRIMAP'])
    print('Unknown region MAD:', res_dict['MAD_TRIMAP'])
    print('Foreground SAD:', res_dict['SAD_FG'])
    print('Background SAD:', res_dict['SAD_BG'])
    print('Gradient:', res_dict['GRAD'])
    print('Connectivity:', res_dict['CONN'])
    return res_dict


if __name__ == '__main__':
    args = get_args()
    evaluate_folder(args)
