import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from imageio import imread, imwrite

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import models


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--glob', help='Input mode 1: path to input images')
    parser.add_argument('--txt', help='Input mode 2: path to image name txt file')
    parser.add_argument('--root', help='Input mode 2: path to input images root')
    parser.add_argument('--pth', help='path to dumped .pth file', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rgb_mean', default=[123.675, 116.28, 103.53], type=float, nargs=3)
    parser.add_argument('--rgb_std', default=[58.395, 57.12, 57.375], type=float, nargs=3)
    parser.add_argument('--base_height', default=512, type=int)
    parser.add_argument('--crop_black', default=80/512, type=float)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Prepare all input rgb paths
    if args.glob is not None:
        assert args.txt is None and args.root is None
        rgb_paths = sorted(glob.glob(args.glob))
    else:
        with open(args.txt) as f:
            rgb_paths = [os.path.join(args.root, line.strip().split()[0]) for line in f]
    for path in rgb_paths:
        assert os.path.isfile(path) or os.path.islink(path)
    print('%d images in total.' % len(rgb_paths))

    # Load trained checkpoint
    print('Loading checkpoint...', end='', flush=True)
    net, args_model = utils.load_trained_model(args.pth)
    net = net.eval().to(args.device)
    print('done')

    # Inference on all images
    for path in tqdm(rgb_paths):
        k = os.path.split(path)[1][:-4]
        rgb_np = imread(path)[..., :3]

        with torch.no_grad():
            # Prepare 1,3,H,W input tensor
            input_dict = {
                'rgb': torch.from_numpy(rgb_np.transpose(2, 0, 1)[None].astype(np.float32)),
            }
            input_dict = utils.preprocess(input_dict, args)  # Normalize and cropping
            input_dict['filename'] = k

            # Call network interface for estimated HV map
            infer_dict = net.infer_HVmap(input_dict, args)

        # Dump results
        for name, v in infer_dict.items():
            if name == 'h_planes':
                imwrite(os.path.join(args.outdir, k + '.h_planes.exr'), v)
            elif name == 'v_planes':
                imwrite(os.path.join(args.outdir, k + '.v_planes.exr'), v)
            elif v.dtype == np.uint8:
                imwrite(os.path.join(args.outdir, k + name + '.png'), v)
            else:
                imwrite(os.path.join(args.outdir, k + name + '.exr'), v)

