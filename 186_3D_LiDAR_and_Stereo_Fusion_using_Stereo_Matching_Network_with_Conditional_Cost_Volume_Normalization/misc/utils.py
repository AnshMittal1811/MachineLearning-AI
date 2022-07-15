"""
Miscellaneous utility functions.
"""
import os
import sys
import shutil
from termcolor import colored
from easydict import EasyDict
import torch


def validate_dir(*dir_name, **kwargs):
    """
    Check and validate a directory
    Args:
        *dir_name (str / a list of str): a directory
        **kwargs:
            auto_mkdir (bool): automatically make directories. Default: True.
        Returns:
            dir_name (str): path to the directory
        Notes:
            1. `auto_mkdir` is performed recursively, e.g. given a/b/c,
               where a/b does not exist, it will create a/b and then a/b/c.
            2. using **kwargs is for future extension.
    """
    # parse argument
    if kwargs:
        auto_mkdir = kwargs.pop('auto_mkdir')
        if kwargs:
            raise ValueError('Invalid arguments: {}'.format(kwargs))
    else:
        auto_mkdir = True

    # check and validate directory
    dir_name = os.path.abspath(os.path.join(*dir_name))
    if auto_mkdir and not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    return dir_name


def setup_workspace(name):
    """ Setup workspace and backup important files """
    workspace = EasyDict()
    workspace.root = validate_dir(name)
    workspace.ckpt = validate_dir(os.path.join(name, 'ckpt'))
    workspace.log = validate_dir(os.path.join(name, 'log'))

    # NOTE: check paths to options.py and train.py
    shutil.copyfile('./misc/options.py', os.path.join(workspace.root, '{}_options.py'.format(name.split('/')[-1])))
    shutil.copyfile('./train.py', os.path.join(workspace.root, '{}_train.py'.format(name.split('/')[-1])))

    return workspace


def save_checkpoint(ckpt_dir, model, optim, scheduler, epoch, global_step):
    """ Save training checkpoints (in iters). """
    states = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch,
        'global_step': global_step
    }
    if scheduler is not None:
        states['scheduler'] = scheduler.state_dict()
    ckpt_path = os.path.join(ckpt_dir, '[ep-{:02d}]giter-{}.ckpt'.format(epoch, global_step))
    torch.save(states, ckpt_path)

    return ckpt_path


def load_checkpoint(model, optim, scheduler, ckpt_path, weight_only=False):
    """ Load checkpoints. """
    states = torch.load(ckpt_path)
    model.load_state_dict(states['model'])
    if not weight_only:
        optim.load_state_dict(states['optim'])
        if scheduler is not None:
            scheduler.load_state_dict(states['scheduler'])

    return states['epoch'], states['global_step']


def disp2depth(disp, img_w):
    """ Convert disparity to depth """
    baseline = 0.54
    width_to_focal = dict()
    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.856
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046 # NOTE: [wrong] assume linear to width 1224
    width_to_focal[1238] = 718.3351

    focal_length = width_to_focal[img_w]
    depth = baseline * focal_length / (disp + 1E-8)
    depth = depth.clamp(max=100.0) # NOTE: clamp to maximum depth as 100 for KITTI
    return depth


class Logger(object):
    """ Logger that can print on terminal and save log to file simultaneously """
    def __init__(self, log_path, mode='w'):
        """ Constructor of Logger
            Args:
                `log_path` (str): full path to log file
        """
        if mode == 'a':
            self._log_fout = open(log_path, 'a')
        elif mode == 'w':
            self._log_fout = open(log_path, 'w')
        else:
            raise ValueError('Invalid mode')

    def write(self, out_str, color='white', end='\n', print_out=True):
        """ Write log
            Args:
                `out_str` (str): string to be printed out and written to log file
        """
        self._log_fout.write(out_str+end)
        self._log_fout.flush()
        if print_out:
            print(colored(out_str, color), end=end)
        sys.stdout.flush()
