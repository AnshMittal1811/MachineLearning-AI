import argparse
from ast import literal_eval
import copy
import logging
import os
from yacs.config import _assert_with_logging, _check_and_coerce_cfg_value_type

from omegaconf import OmegaConf, DictConfig
from .defaults import get_cfg_defaults, CN
from nnutils.model_utils import get_model_name, latest_ckpt


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config-file", default=None, metavar="FILE", help="path to config file")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--ckpt", default=None, metavar="FILE", help="path to config file")

    parser.add_argument("--eval", action='store_true', help="perform evaluation only")
    parser.add_argument("--gpu", type=str, default='1', help="use gpu")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_cfg(args) -> DictConfig:
    ckpt_config = None
    if args.eval and args.ckpt is None:
        ckpt_config = os.path.join(args.ckpt.split('checkpoints')[0], 'hparams.yaml')
        args.ckpt = latest_ckpt(args.ckpt, include_last=True)
    cfg = get_cfg_defaults()
    cfg = merge_cfg(cfg, args.config_file, ckpt_config, args.opts)
    set_model_name_path(args, cfg)

    cfg.freeze()
    cfg = OmegaConf.create(cfg.dump())
    return cfg


def set_model_name_path(args, cfg):
    # set model dir
    if args.eval:
        skip_list = ['EXP', 'GPU', 'TEST.NAME', ]
        cfg.TEST.DIR = cfg.TEST.NAME
        for full_key, v in zip(args.opts[0::2], args.opts[1::2]):
            flag = False
            for skip_word in skip_list:
                if full_key.startswith(skip_word):
                    flag = True
                    break
            if flag:
                continue 
            cfg.TEST.DIR += '_%s%s' % (full_key, str(v))
        # assume ckpt is OUTPUT_DIR/exp/name/checkpoints/latest.ckpt
        cfg.MODEL_SIG = args.ckpt.split('/')[-4] + '/' + args.ckpt.split('/')[-3]
        cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_SIG, 'checkpoints')
    else:
        cfg.MODEL_SIG = os.path.join(cfg.EXP, get_model_name(cfg, args.opts, args.eval, args.config_file))
        cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_SIG, 'checkpoints')


def merge_cfg(cfg, *args):
    for each in args:
        if isinstance(each, str):
            with open(each, 'r') as fp:
                each = CN.load_cfg(fp)
            _merge_a_into_b(each, cfg, cfg, [], )
        elif isinstance(each, CN):
            _merge_a_into_b(each, cfg, cfg, [], )
        elif isinstance(each, list):
            cfg.merge_from_list(each)
        elif each is None:
            continue
        else:
            raise NotImplementedError('type %s' % type(each))
    return cfg




def _merge_a_into_b(a, b, root, key_list=[], skip_list=[]):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    _assert_with_logging(
        isinstance(a, CN),
        "`a` (cur type {}) must be an instance of {}".format(type(a), CN),
    )
    _assert_with_logging(
        isinstance(b, CN),
        "`b` (cur type {}) must be an instance of {}".format(type(b), CN),
    )

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])

        v = copy.deepcopy(v_)
        v = b._decode_cfg_value(v)

        if k in b:
            if root.key_is_deprecated(full_key):
                continue
            if full_key in skip_list:
                print('skip ', full_key)
                continue

            v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
            # Recursively merge dicts
            if isinstance(v, CN):
                try:
                    _merge_a_into_b(v, b[k], root, key_list + [k], skip_list)
                except BaseException:
                    raise
            else:
                b[k] = v
        elif b.is_new_allowed():
            b[k] = v
        else:
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            else:
                b[k] = v
                print("Non-existent config key: {}".format(full_key))
                # raise KeyError("Non-existent config key: {}".format(full_key))


def merge_list_into_b(cfg_list, d):
    root = d
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        d = root
        key_list = full_key.split(".")
        for subkey in key_list[:-1]:
            d = d[subkey]
        subkey = key_list[-1]
        def decode_value(value):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
            return value
        value = decode_value(v)
        d[subkey] = value
    return root

