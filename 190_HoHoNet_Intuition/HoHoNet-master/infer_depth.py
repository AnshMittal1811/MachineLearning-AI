import os, sys, time, glob
import argparse
import importlib
from tqdm import tqdm
from imageio import imread, imwrite
import torch
import numpy as np

from lib.config import config, update_config


if __name__ == '__main__':

    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--pth', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--inp', required=True)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    device = 'cuda' if config.cuda else 'cpu'

    # Parse input paths
    rgb_lst = glob.glob(args.inp)
    if len(rgb_lst) == 0:
        print('No images found')
        import sys; sys.exit()

    # Init model
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs)
    net.load_state_dict(torch.load(args.pth, map_location=device))
    net = net.eval().to(device)

    # Run inference
    with torch.no_grad():
        for path in tqdm(rgb_lst):
            rgb = imread(path)
            x = torch.from_numpy(rgb).permute(2,0,1)[None].float() / 255.
            if x.shape[2:] != config.dataset.common_kwargs.hw:
                x = torch.nn.functional.interpolate(x, config.dataset.common_kwargs.hw, mode='area')
            x = x.to(device)
            pred_depth = net.infer(x)
            if not torch.is_tensor(pred_depth):
                pred_depth = pred_depth.pop('depth')

            fname = os.path.splitext(os.path.split(path)[1])[0]
            imwrite(
                os.path.join(args.out, f'{fname}.depth.png'),
                pred_depth.mul(1000).squeeze().cpu().numpy().astype(np.uint16)
            )

