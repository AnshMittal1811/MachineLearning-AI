import os
from yacs.config import CfgNode as CN

config = CN()

config.ckpt_root = 'ckpt'
config.cuda = True
config.cuda_benchmark = True
config.num_workers = 8

config.dataset = CN()
config.dataset.name = 'PanoCorBonDataset'
config.dataset.common_kwargs = CN(new_allowed=True)
config.dataset.train_kwargs = CN(new_allowed=True)
config.dataset.valid_kwargs = CN(new_allowed=True)

config.training = CN()
config.training.epoch = 300
config.training.batch_size = 4
config.training.save_every = 100
config.training.optim = 'Adam'
config.training.optim_lr = 0.0001
config.training.optim_betas = (0.9, 0.999)
config.training.weight_decay = 0.0
config.training.wd_group_mode = 'bn and bias'
config.training.optim_milestons = [0.5, 0.9]
config.training.optim_gamma = 0.2
config.training.optim_poly_gamma = -1.0
config.training.fix_encoder_bn = False

config.model = CN()
config.model.file = 'lib.model.HorizonNet'
config.model.modelclass = 'HorizonNet'
config.model.kwargs = CN(new_allowed=True)


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

def infer_exp_id(cfg_path):
    cfg_path = cfg_path.split('config/')[-1]
    if cfg_path.endswith('.yaml'):
        cfg_path = cfg_path[:-len('.yaml')]
    return '_'.join(cfg_path.split('/'))

