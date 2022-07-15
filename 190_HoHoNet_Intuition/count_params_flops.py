import os
import argparse
import importlib
from tqdm import tqdm, trange
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile, clever_format

from lib.config import config, update_config


if __name__ == '__main__':

    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    # Init global variable
    device = 'cuda' if config.cuda else 'cpu'
    if config.cuda and config.cuda_benchmark:
        torch.backends.cudnn.benchmark = True

    # Init network
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs).to(device)
    net.eval()

    # testing
    layers = net
    inputs = [torch.randn(1, 3, 512, 1024).to(device)]
    with torch.no_grad():
        flops, params = profile(layers, inputs)
    print(f'input :', [v.shape for v in inputs])
    print(f'flops : {flops/(10**9):.2f} G')
    print(f'params: {params/(10**6):.2f} M')

    import time
    fps = []
    with torch.no_grad():
        layers(inputs[0])
        for _ in range(50):
            eps_time = time.time()
            layers(inputs[0])
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
    print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')

