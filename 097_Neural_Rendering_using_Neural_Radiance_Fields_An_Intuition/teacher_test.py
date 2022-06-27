import os
import numpy as np
import torch
from omegaconf import DictConfig
import hydra
from mnh.dataset_replica import dataset_to_depthpoints
from mnh.dataset import load_datasets
from mnh.utils_vedo import visualize_geometry
from mnh.stats import StatsLogger
from mnh.utils import *
from mnh.utils_vedo import get_vedo_alpha_plane
from teacher_forward import *

CURRENT_DIR = os.path.realpath('.')
CONFIG_DIR = os.path.join(CURRENT_DIR, 'configs')
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')

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
    datasets = {
        'train': train_dataset,
        'valid': valid_dataset
    }

    model = get_model_from_config(cfg)
    model.to(device)
    model.eval()

    # load checkpoints
    checkpoint_path = os.path.join(CHECKPOINT_DIR, cfg.checkpoint.teacher)
    if os.path.isfile(checkpoint_path):
        print('Load from checkpoint: {}'.format(checkpoint_path))
        loaded_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(loaded_data['model'])
    else:
        # initialize plane position, rotation and size
        print('[Init] initialize plane geometry ...')
        points = train_dataset.dense_points.to(device)
        print('#points= {}'.format(points.size(0)))
        if 'replica' in cfg.data.path:
            model.plane_geo.initialize_with_box(
                points, 
                lrf_neighbors=cfg.model.init.lrf_neighbors,
                wh=cfg.model.init.wh,
                box_factor=cfg.model.init.box_factor, 
                random_rate=cfg.model.init.random_rate,
            )
        else:
            model.plane_geo.initialize(
                points,
                lrf_neighbors=cfg.model.init.lrf_neighbors,
                wh=cfg.model.init.wh,
            )
        del points 
        torch.cuda.empty_cache()
    
    if cfg.model.accelerate.bake == True:
        model.bake_planes_alpha()
    output_dir = os.path.join(CURRENT_DIR, 'output_images', cfg.name, 'teacher')
    os.makedirs(output_dir, exist_ok=True)
    
    print('Test [{}] ...'.format(cfg.test.mode))
    if cfg.test.mode == 'test_model':
        print('- Parameter number: {}'.format(parameter_number(model)))
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.optimizer.lr
        )
        train_data = train_dataset[0]
        train_stats, _ = forward_pass(train_data, model, device, cfg, optimizer, training=True)
        print(train_stats)
        print('- Train: forward + backprop')
        valid_data = valid_dataset[0]
        valid_stats, _ = forward_pass(valid_data, model, device, cfg, training=False)
        print(valid_stats)
        print('- Validation: forward')
        print('- Image inference FPS: {:.3f}'.format(valid_stats['FPS']))

    if cfg.test.mode == 'evaluate':
        stats_logger= StatsLogger()
        model.eval()
        for split, dataset in datasets.items():
            for i in range(len(dataset)):
                data = dataset[i]
                stats, _ = forward_pass(data, model, device, cfg)
                stats_logger.update(split, stats)
            stats_logger.print_info(split)

    if cfg.test.mode == 'render':
        folder = {}
        splits = ['valid']
        keys = ['color', 'depth']
        for key in keys:
            folder[key] = {}
            for split in splits:
                if cfg.test.folder == '':
                    path = os.path.join(output_dir, key, split)
                else:
                    path = os.path.join(output_dir, cfg.test.folder, key, split)
                os.makedirs(path, exist_ok=True)
                folder[key][split] = path
        
        stats_logger = StatsLogger()
        for split in splits:
            print('saving [{}] images ...'.format(split))
            dataset = datasets[split]
            depths_all = []
            for i in range(len(dataset)):
                data = dataset[i]
                stats, images = forward_pass(data, model, device, cfg)
                
                for label in ['gt', 'pred']:
                    img = images['color_{}'.format(label)]
                    img = tensor2Image(img)
                    path = os.path.join(folder['color'][split], '{}-{:0>5}-{}.png'.format(split, i, label))
                    img.save(path)
                depth = images['depth_pred']
                depths_all.append(depth)
                stats_logger.update(split, stats)
            
            depths_all = torch.stack(depths_all, dim=0)
            depths_all = to_numpy(depths_all)
            path = os.path.join(folder['depth'][split], 'depth.npy')
            np.save(path, depths_all)

            stats_logger.print_info(split)
            avg_time = stats_logger.stats[split]['time'].get_mean()
            print('FPS: {:.3f}'.format(1 / avg_time))

    if cfg.test.mode == 'geometry':
        visualize_geometry(
            train_dataset.dense_points,
            model.plane_geo,
            r=cfg.test.vis.r,
            c=cfg.test.vis.c,
            alpha=cfg.test.vis.alpha
        )
        
if __name__ == '__main__':
    main()