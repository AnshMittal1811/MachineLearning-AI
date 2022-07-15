import os
import importlib
import Imath
import OpenEXR
from imageio import imread
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(args_model):
    # Import the Model class
    Model_module_path = '.'.join(args_model['model_setting']['model'].split('.')[:-1])
    Model_name = args_model['model_setting']['model'].split('.')[-1]
    Model = getattr(importlib.import_module(Model_module_path), Model_name)

    # Instanciation
    model_kwargs = dict(args_model['model_kwargs'])
    model_kwargs['backbone_kwargs'] = args_model['backbone_kwargs']
    net = Model(**model_kwargs)
    return net

def read_depth(path, div):
    if path.endswith('png'):
        depth = imread(path)
    elif path.endswith('exr'):
        f = OpenEXR.InputFile(path)
        dw = f.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        depth = np.frombuffer(f.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
        depth = depth.reshape(size[1], size[0])
        f.close()
    else:
        raise NotImplementedError()
    dontcare = (depth <= 0) | (depth >= 65535)
    depth = (depth / div).astype(np.float32)
    return depth, dontcare

def read_h_planes(path):
    assert path.endswith('exr')
    f = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    h_planes = np.frombuffer(f.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
    h_planes = h_planes.reshape(size[1], size[0])
    f.close()
    return h_planes

def read_v_planes(path):
    assert path.endswith('exr')
    f = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    v_planes = np.stack([
        np.frombuffer(f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
        for ch in 'RGB'
    ], -1)
    v_planes = v_planes.reshape(size[1], size[0], 3)
    f.close()
    return v_planes

def preprocess(input_dict, args):
    for k, v in input_dict.items():
        input_dict[k] = v.to(args.device)

    # Normalize RGB
    rgb_mean = torch.FloatTensor(args.rgb_mean).reshape(1,3,1,1).to(args.device)
    rgb_std = torch.FloatTensor(args.rgb_std).reshape(1,3,1,1).to(args.device)
    input_dict['rgb'] = (input_dict['rgb'] - rgb_mean) / rgb_std

    # Scale
    for k, v in input_dict.items():
        if v.shape[2] != args.base_height:
            H = args.base_height
            input_dict[k] = F.interpolate(v, size=[H, H*2], mode='nearest')

    # Crop top-down black region
    crop = int(args.crop_black * input_dict['rgb'].shape[2])
    for k, v in input_dict.items():
        input_dict[k] = v[:, :, crop:-crop]

    # For later error proof
    input_dict['preprocessed'] = True
    return input_dict

def generate_worker_init_fn(args):
    def worker_init_fn(worker_id):
        new_worker_seed = args.seed + worker_id + args.cur_epoch * args.num_workers
        np.random.seed(new_worker_seed)
    return worker_init_fn

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [
        dict(params=[param for param in group_decay if param.requires_grad]),
        dict(params=[param for param in group_no_decay if param.requires_grad], weight_decay=.0)
    ]

def save_model(net, path, args_model):
    torch.save(OrderedDict({
        'state_dict': net.state_dict(),
        'args_model': args_model,
    }), path)

def load_trained_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    args_model = checkpoint['args_model']

    Model_module_path = '.'.join(args_model['model_setting']['model'].split('.')[:-1])
    Model_name = args_model['model_setting']['model'].split('.')[-1]
    Model = getattr(importlib.import_module(Model_module_path), Model_name)
    net = Model(backbone_kwargs=args_model['backbone_kwargs'], **args_model['model_kwargs'])
    net.load_state_dict(checkpoint['state_dict'])
    return net, args_model
