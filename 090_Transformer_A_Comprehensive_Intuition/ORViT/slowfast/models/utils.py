# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from collections import OrderedDict
import os
import math

import slowfast.utils.logging as logging
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

_logger = logger = logging.get_logger(__name__)


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        logger.info(f"min width {min_width}")
        logger.info(f"width {width} divisor {divisor}")
        logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


# Load
def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    names = ['q', 'k', 'v']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            # v = v.expand((v.shape[0], 3, 3, patch_size, patch_size))
            k = 'patch_embed.proj.weight'
        elif k == 'head.weight':
            k = 'head.projection.weight'
        elif k == 'head.bias':
            k = 'head.projection.bias'
        elif 'qkv' in k:
            # split
            for a, new_v in zip(names, v.chunk(3,dim=0)):
                new_k = k.replace('qkv', a)
                out_dict[new_k]=new_v
            continue
        out_dict[k] = v
    return out_dict

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=_conv_filter, img_size=224, num_frames=8, num_patches=196, attention_type='divided_space_time', pretrained_model="", strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    if len(pretrained_model) == 0:
       state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    else:
       try:
         state_dict = load_state_dict(pretrained_model)['model']
       except:
         state_dict = load_state_dict(pretrained_model)


    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            _logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight


    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    
    elif num_classes != state_dict[classifier_name + '.weight'].size(0):
        #print('Removing the last fully connected layer due to dimensions mismatch ('+str(num_classes)+ ' != '+str(state_dict[classifier_name + '.weight'].size(0))+').', flush=True)
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False


    ## Resizing the positional embeddings in case they don't match
    if num_patches + 1 != state_dict['pos_embed'].size(1):
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed

    ## Resizing time embeddings in case they don't match
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'].size(1):
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=(num_frames), mode='nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)

    if True: #state_dict['patch_embed.proj.weight'].shape[-1] != model.state_dict()['patch_embed.proj.weight'].shape[-1]:
        ksize = model.state_dict()['patch_embed.proj.weight'].shape[-3:]
        kernel = state_dict['patch_embed.proj.weight'] # d d T H W
        d1, d2,  _, _ = kernel.shape
        new_kernel = F.interpolate(kernel, size=ksize[1:], mode='nearest')
        new_kernel = new_kernel.unsqueeze(2).expand(-1, -1, ksize[0], -1, -1) # d1 d2 T H W
        state_dict['patch_embed.proj.weight'] = new_kernel
    state_dict['pos_embed_class'] = state_dict['pos_embed'][:, [0]]
    state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:]
    # Rename
    RENAME_KEYS = [
        ('pos_embed', 'pos_embed_spatial'),
        # (''),
    ]
    for old, new in RENAME_KEYS:
        v = state_dict[old]
        del state_dict[old]
        state_dict[new] = v

    ## Loading the weights
    model.load_state_dict(state_dict, strict=False)


