import os
import glob
import json
import argparse
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import config, update_config, infer_exp_id


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--pth', help='path to load saved checkpoint.')
    parser.add_argument('--img_glob', required=True)
    parser.add_argument('--output_dir', required=True)
    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=int,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Init setting
    update_config(config, args)
    if not args.pth:
        exp_id = infer_exp_id(args.cfg)
        exp_ckpt_root = os.path.join(config.ckpt_root, exp_id)
        args.pth = sorted(glob.glob(os.path.join(exp_ckpt_root, '*pth')))[-1]
        print(f'--pth is not given.  Auto infer the pth={args.pth}')
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    # Prepare image to processed
    paths = sorted(glob.glob(args.img_glob))
    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # Prepare the trained model
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs)
    net.load_state_dict(torch.load(args.pth))
    net = net.to(device).eval()

    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)

    # Inferencing
    with torch.no_grad():
        for i_path in tqdm(paths, desc='Inferencing'):
            k = os.path.split(i_path)[-1][:-4]

            # Load image
            img_pil = Image.open(i_path)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255]).to(device)

            # Inferenceing corners
            net.fname = k
            cor_id = net.infer(x)['cor_id']

            # Output result
            with open(os.path.join(args.output_dir, k + '.txt'), 'w') as f:
                for x, y in cor_id:
                    f.write('%d %d\n' % (x, y))

