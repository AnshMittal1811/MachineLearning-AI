# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import torch
from collections import OrderedDict

from lib.config import config
from lib.utils.imports import import_file


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        # logger.info(
        #     log_str_template.format(
        #         key,
        #         max_size,
        #         key_old,
        #         max_size_loaded,
        #         tuple(loaded_state_dict[key_old].shape),
        #     )
        # )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def remove_dict(model_dict, keyword):
    keys = set(model_dict.keys())
    for key in keys:
        if keyword in key:
            del model_dict[key]
    return model_dict

def fix_weights(model, keyword):
    for key, value in model.named_parameters():
        if keyword in key:
            # print(key)
            value.requires_grad = False

def switch_train(model):
    model.train()
    if config.MODEL.FIX2D:
        model.encoder2d.eval()
        if config.MODEL.DEPTH2D.USE:
            model.depth2d.eval()
        if config.MODEL.INSTANCE2D.USE:
            model.instance2d.eval()

    if config.MODEL.FRUSTUM3D.FIX:
        model.frustum3d.eval()

    if config.MODEL.DEPTH2D.FIX:
        model.depth2d.eval()

    if config.MODEL.INSTANCE2D.USE and config.MODEL.INSTANCE2D.FIX:
        model.instance2d.eval()

def switch_test(model):
    model.eval()

def count_parameters(model, check_grad=False):
    if check_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    #if config.PRETRAIN:
    #    import ipdb
    #    ipdb.set_trace()
    #    loaded_state_dict = remove_dict(loaded_state_dict, 'mask3d')
    #    model_state_dict.update(loaded_state_dict)
    #    model.load_state_dict(model_state_dict)
    #else:
    model.load_state_dict(model_state_dict)
