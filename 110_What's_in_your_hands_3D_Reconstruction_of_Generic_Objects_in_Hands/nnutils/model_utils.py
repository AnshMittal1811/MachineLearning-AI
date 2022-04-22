import importlib
import logging
import numpy as np
import os
import os.path as osp
import glob

import torch
import torch.nn as nn
import pytorch_lightning as pl


def latest_ckpt(prop, include_last=False, till=-1):
    if not prop.endswith('.ckpt'):
        if include_last:
            ckpt = os.path.join(prop, 'checkpoints', 'last.ckpt')
            if os.path.exists(ckpt):
                return ckpt
        ckpt_list = glob.glob(os.path.join(prop, 'checkpoints', 'epoch*.ckpt'))
        print(ckpt_list)
        epoch_list = [int(os.path.basename(e)[len('epoch='):].split('-step')[0]) for e in ckpt_list]
        last_ckpt = os.path.join(prop, 'checkpoints/last.ckpt')
        if len(epoch_list) == 0 and os.path.exists(last_ckpt):
            return last_ckpt
        if len(epoch_list) == 0:
            return None
        inds = np.argmax(epoch_list)
        return ckpt_list[inds]


def get_model_name(cfg,  cli_args, eval, config_file):
    if eval:
        name = os.path.basename(cfg.MODEL_SIG)
    else:
        # name = '%s' % (cfg.DB.NAME)
        # name += '_%s' % (cfg.MODEL.NAME)
        name = '%s' % (osp.basename(config_file).split('.')[0])
    
        skip_list = ['EXP', 'GPU',
                     'TEST.NAME']
        for full_key, v in zip(cli_args[0::2], cli_args[1::2]):
            if full_key in skip_list:
                continue
            name += '_%s%s' % (full_key, str(v))

    return name
    

def load_model(cfg, ckpt_dir, ckpt_epoch) -> nn.Module:

    Model = getattr(importlib.import_module(".ihoi", "models"), cfg.MODEL.NAME)

    model = Model(cfg)
    ckpt = osp.join(ckpt_dir, 'checkpoints', '%s.ckpt' % ckpt_epoch)
    logging.info('load from %s' % ckpt)
    ckpt = torch.load(ckpt)['state_dict']
    load_my_state_dict(model, ckpt)
    
    model.eval()
    model.cuda()
    return model
            

def load_my_state_dict(model: torch.nn.Module, state_dict, lambda_own=lambda x: x):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        own_name = lambda_own(name)
        # own_name = '.'.join(name.split('.')[1:])
        if own_name not in own_state:
            logging.warn('Not found in checkpoint %s %s' % (name, own_name))
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[own_name].size():
            logging.warn('size not match %s %s %s' % (
                name, str(param.size()), str(own_state[own_name].size())))
            continue
        own_state[own_name].copy_(param)



def to_cuda(data, device='cuda'):
    new_data = {}
    for key in data:
        if hasattr(data[key], 'cuda'):
            new_data[key] = data[key].to(device)
        else:
            new_data[key] = data[key]
    return new_data