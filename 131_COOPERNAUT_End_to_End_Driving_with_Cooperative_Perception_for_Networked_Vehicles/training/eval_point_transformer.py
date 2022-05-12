import os
import ray
import time
import glob
import tqdm
import wandb
import torch
import signal
import subprocess
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch import optim
from copy import deepcopy
from models import PointTransformer, CooperativePointTransformer
from .logger import Logger
from utils.point_transformer_loader import get_data_loader


def _numpy(x):
    return x.detach().cpu().numpy()

def evaluate(config):
    if config.cpt:
        model = CooperativePointTransformer(config).to(config.device)
    else:
        model = PointTransformer(config).to(config.device)
    dataset = get_data_loader(config)
    if config.finetune is not None:
        print("Loading Model:", config.finetune)
        model.load_state_dict(torch.load(config.finetune))
    model.eval()
    torch.cuda.empty_cache()
    dataset = get_data_loader(config)
    moving_loss = []
    for batch_data in tqdm.tqdm(dataset):
        bev_rgb, ego_lidar, ego_speed, ego_brake, ego_has_plan, ego_command, ego_control, other_lidar, other_speed, other_transform, ego_transform, num_valid_neighbors = batch_data

        # Convert tensors
        ego_lidar = ego_lidar.float().to(config.device)
        ego_speed = ego_speed.float().to(config.device)
        ego_brake = ego_brake.float().to(config.device)
        ego_throttle = ego_control[:,0].float().to(config.device)
        ego_steer = ego_control[:,2].float().to(config.device)
        ego_has_plan = ego_has_plan.byte().to(config.device)
        ego_command = ego_command.long().to(config.device)
        ego_control = ego_control.float().to(config.device)
        ego_transform = ego_transform.float().to(config.device)
        num_valid_neighbors = num_valid_neighbors.float().to(config.device)
        if config.cpt:
            other_lidar = other_lidar.float().to(config.device)
            other_speed = other_speed.float().to(config.device)
            other_transform = other_transform.float().to(config.device)
            ego_transform = ego_transform.float().to(config.device)
            num_valid_neighbors = num_valid_neighbors.float().to(config.device)
            pred_throttle, pred_brake, pred_steer = model(ego_lidar, ego_speed, other_lidar, other_transform)
        else:
            pred_throttle, pred_brake, pred_steer = model(ego_lidar, ego_speed)

        coeff_throttle = 1.0
        coeff_brake = 1.0
        coeff_steer = 1.0
        throttle_loss = F.l1_loss(pred_throttle, ego_throttle)
        brake_loss = F.l1_loss(pred_brake, ego_brake)
        steer_loss = F.l1_loss(pred_steer, ego_steer)
        loss = coeff_throttle * throttle_loss  + coeff_brake * brake_loss + coeff_steer * steer_loss
            
        moving_loss.append(_numpy(loss))
        print(_numpy(loss))

    return float(np.array(moving_loss).mean())


if __name__ == '__main__':
    
    import argparse
    import tqdm
    parser = argparse.ArgumentParser()

    # Dataloader configs
    parser.add_argument('--data', required=True, help='path to the dataset. it should contains folders named 0, 1, etc. check: https://drive.google.com/drive/u/1/folders/1xmZiu9yiFw2IdQETKtA4KyiXPWh-2MIH')
    parser.add_argument('--daggerdata', default=None)
    parser.add_argument('--ego-only', action='store_true', help='only return data of one agent', default=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-workers', type=int, default=3,
                        help='number of ray workers/carla instances.')
    parser.add_argument('--num-dataloader-workers', type=int, default=8, help='number of pytorch dataloader workers. set it to zero for debugging')
    parser.add_argument('--use-lidar', action='store_true', help='Lidar information as the 4th data returned')
    parser.add_argument('--visualize', action='store_true', help='wether we are visualizing')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--num-commands', type=int, default=6)
    parser.add_argument('--frame-stack', type=int, default=1, help='num of frames that are stacked')
    parser.add_argument('--max_num_neighbors', type=int, default=2, help='max number of neighbors that we consider')
    parser.add_argument('--fullpc', action="store_true",
                        help='Enable Full Point cloud sharing with 1000X bandwidth')
    # Model configs
    parser.add_argument('--num-hidden', type=int, default=256)
    parser.add_argument('--npoints', type=int, default=4096)
    parser.add_argument('--nblocks', type=int, default=2)
    parser.add_argument('--nneighbor', type=int, default=16)
    parser.add_argument('--num_output', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--transformer_dim', type=int, default=512) 
    parser.add_argument('--uniform', action='store_true')
    # Traninig configs
    parser.add_argument('--project', default='autocast')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--init-lr', type=float, default=3e-4)
    parser.add_argument('--num-steps-per-log', type=int, default=10)
    parser.add_argument('--checkpoint-frequency', type=int, default=25)
    parser.add_argument('--finetune',type=str, default=None)

    #Cooperative Model
    parser.add_argument('--cpt', action='store_true', help='whether to use a cooperative model for multi-car')
    #Early Fusion Model
    parser.add_argument('--earlyfusion', action='store_true', help='whether to use an early fusion model')
    # DAgger configs
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--sampling-frequency', type=int, default=25)
    parser.add_argument('--benchmark_config',type=str,default="benchmark/scene6.json")
    parser.add_argument('--bgtraffic', type=int, default=None,
                        help='Set the amount of background traffic in scenario')
    parser.add_argument('--resample-config', type=str, default='fixed',
                        help='how do we treat the config parameters')
    parser.add_argument('--num-config', type=int, default=None,
                        help='if not None, sample a subset of the configs to run')

    config = parser.parse_args()
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        val_loss = evaluate(config)

    print("Validation Loss:", val_loss)
