import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle
import copy
from mnh.dataset import load_datasets
from mnh.model_teacher import *
from mnh.stats import StatsLogger
from mnh.utils import *
from mnh.utils_model import freeze_model
import teacher_forward
from experts_forward import *

CURRENT_DIR = os.path.realpath('.')
CONFIG_DIR = os.path.join(CURRENT_DIR, 'configs')
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


@hydra.main(config_path=CONFIG_DIR)
def main(cfg: DictConfig):
    # Set random seed for reproduction
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set device for training
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cfg.cuda))
    else:
        device = torch.device('cpu')

    # set DataLoader objects
    train_dataset, valid_dataset = load_datasets(os.path.join(CURRENT_DIR, cfg.data.path), cfg)
    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=False)
    valid_loader = DataLoader(valid_dataset, collate_fn=lambda x: x, shuffle=False)

    teacher = teacher_forward.get_model_from_config(cfg)
    teacher.to(device)
    model = get_model_from_config(cfg)
    model.to(device)

    # load checkpoints
    stats_logger = None
    optimizer_state = None   
    start_epoch = 0 

    checkpoint_teacher = os.path.join(CHECKPOINT_DIR, cfg.checkpoint.teacher)
    pretrained_teacher = os.path.isfile(checkpoint_teacher)
    if pretrained_teacher: 
        print('Load teacher from checkpoint: {}'.format(checkpoint_teacher))
        loaded_data = torch.load(checkpoint_teacher, map_location=device)
        teacher.load_state_dict(loaded_data['model'])
    else:
        print('WARNING: no pretrained weight for teacher network')
    
    checkpoint_experts = os.path.join(CHECKPOINT_DIR, cfg.checkpoint.experts)
    if cfg.train.resume and os.path.isfile(checkpoint_experts):
        print('Resume training from checkpoint: {}'.format(checkpoint_experts))
        loaded_data = torch.load(checkpoint_experts, map_location=device)
        model.load_state_dict(loaded_data['model'])
        stats_logger = pickle.loads(loaded_data['stats'])
        start_epoch = stats_logger.epoch
        optimizer_state = loaded_data['optimizer']
    else:
        if pretrained_teacher:
            print('[Init] Copy plane geometry from teacher ...')
            model.plane_geo = copy.deepcopy(teacher.plane_geo)
        else:
            print('[Init] Initialize plane geometry')
            points = train_dataset.dense_points.to(device)
            model.plane_geo.initialize(
                points, 
                lrf_neighbors=cfg.model.init.lrf_neighbors,
                wh=cfg.model.init.wh,
            )
            del points

    if cfg.train.freeze_geometry:
        print('Freeze plane geometry')
        freeze_model(model.plane_geo)
    
    # set optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.optimizer.lr
    )
    if optimizer_state != None:
        optimizer.load_state_dict(optimizer_state)
        optimizer.last_epoch = start_epoch
    
    def lr_lambda(epoch):
        return cfg.optimizer.lr_scheduler_gamma ** (
            epoch / cfg.optimizer.lr_scheduler_step_size
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    # set StatsLogger, WandbLogger objects
    if stats_logger == None:
        stats_logger = StatsLogger()
    
    img_folder = os.path.join(CURRENT_DIR, 'output_images', cfg.name, 'experts', 'output')
    os.makedirs(img_folder, exist_ok=True)
    print('[Traing Experts]')

    epoch_total = cfg.train.epoch.distill + cfg.train.epoch.finetune
    for epoch in range(start_epoch, epoch_total):
        model.train()
        stats_logger.new_epoch()

        for i, data in enumerate(train_loader):
            data = data[0]
            if epoch < cfg.train.epoch.distill:
                train_stats = learn_from_teacher(
                    data, 
                    model, 
                    teacher,
                    device, 
                    cfg, 
                    optimizer
                )
            else:
                train_stats, _ = forward_pass(
                    data, 
                    model,
                    device,
                    cfg, 
                    optimizer,
                    training=True,
                )
            stats_logger.update('train', train_stats)
        
        stats_logger.print_info('train')
        lr_scheduler.step()

        # Checkpoint
        if (epoch+1) % cfg.train.epoch.checkpoint == 0:
            print('store checkpoints ...')
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stats': pickle.dumps(stats_logger)
            }
            torch.save(checkpoint, checkpoint_experts)

        # validation
        if (epoch+1) % cfg.train.epoch.validation == 0:
            model.eval()
            for i, data in enumerate(valid_loader):
                data = data[0]
                valid_stats, valid_images = forward_pass(
                    data, 
                    model,
                    device,
                    cfg,
                    training=False,
                )
                stats_logger.update('valid', valid_stats)

                for key, img in valid_images.items():
                    if 'depth' in key:
                        img = img / img.max()
                    img = tensor2Image(img)
                    path = os.path.join(img_folder, 'valid-{:0>5}-{}.png'.format(i, key))
                    img.save(path)

            stats_logger.print_info('valid')

if __name__ == '__main__':
    main()