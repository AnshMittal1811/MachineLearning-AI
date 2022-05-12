import time
import argparse

from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tqdm
import MinkowskiEngine as ME
import torch.nn.functional as F
import glob
import os
import sys
sys.path.append('../')
sys.path.append('../../')
from models import V2VNet
from utils.voxel_v2v_loader import get_data_loader
import pygame
from AutoCastSim.AVR import Utils
    
def visualize_dataset(config):
    dataset = get_data_loader(config)
    model = V2VNet(config).to(config.device)
    
    model.eval()
    model_path = os.path.join(config.log_dir, config.model)
    print("loading model: %s"%model_path)
    model.load_state_dict(torch.load(model_path))
    val_loss = []
    for batch_data in dataset:
        bev_rgb, bev, ego_future, ego_speed, ego_brake, ego_has_plan, ego_command, ego_control, other_bev, other_speed, other_transform, ego_transform = batch_data
        bev = bev.float().to(config.device)
        ego_future = ego_future.float().to(config.device)
        ego_speed = ego_speed.float().to(config.device)
        ego_brake = ego_brake.float().to(config.device)
        ego_throttle = ego_control[:,0].float().to(config.device)
        ego_steer = ego_control[:,2].float().to(config.device)

        ego_has_plan = ego_has_plan.byte().to(config.device)
        ego_command = ego_command.long().to(config.device)
        ego_control = ego_control.float().to(config.device)
        other_bev = other_bev.float().to(config.device)
        other_speed = other_speed.float().to(config.device)
        other_transform = other_transform.float().to(config.device)
        ego_transform = ego_transform.float().to(config.device)

        pred_throttle, pred_brake, pred_steer = model(bev, ego_speed, ego_command,other_bev, other_speed, other_transform, ego_transform)
        coeff_throttle = 0.5
        coeff_brake = 1.0
        coeff_steer = 1.0
            
        throttle_loss = F.l1_loss(pred_throttle, ego_throttle)
        brake_loss = F.l1_loss(pred_brake, ego_brake)
        steer_loss = F.l1_loss(pred_steer, ego_steer)

        loss = coeff_throttle * throttle_loss  + coeff_brake * brake_loss + coeff_steer * steer_loss
        print(" Loss:", loss, "Throttle Loss:", throttle_loss.item(), "Brake Loss:", brake_loss.item(), "Steer Loss:", steer_loss.item())#, " future_loss:", future_loss, "brake_loss:", brake_loss)
        print(pred_throttle, ego_throttle)
        print(pred_brake, ego_brake)
        print(pred_steer, ego_steer)
        val_loss.append(loss.item())

    print("Validation Loss Mean: ",np.mean(val_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset.
    parser.add_argument('--dataset_dir', default='/raid0/dian/carla_0.9.6_data')

    # Dataloader configs
    parser.add_argument('--data', required=True, help='path to the dataset. it should contains folders named 0, 1, etc. check: https://drive.google.com/drive/u/1/folders/1xmZiu9yiFw2IdQETKtA4KyiXPWh-2MIH')
    parser.add_argument('--ego-only', action='store_true', help='only return data of one agent')
    parser.add_argument('--use-lidar', action='store_true', help='Lidar information as the 4th data returned')
    parser.add_argument('--visualize', action='store_true', help='wether we are visualizing')
    parser.add_argument('--use-speed', action='store_true', help='wether to use speed in the model')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-dataloader-workers', type=int, default=0, help='number of pytorch dataloader workers. set it to zero for debugging')
    parser.add_argument('--model', type=str, default='model-500.th', help='path to the model to be loaded')
    parser.add_argument('--model_name', type=str, default='MultiInput', help='specific model to be loaded')
    parser.add_argument('--log_dir', type=str, default='../../training', help='dir that stores model')
    parser.add_argument('--shared', type=bool, default=False, help='whether use shared lidar data')
    parser.add_argument('-T', type=int, default=10, help='num of waypoints to predict')
    parser.add_argument('--num-commands', type=int, default=6)
    parser.add_argument('--num-hidden', type=int, default=256)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--frame-stack',type=int,default=1, help='number of frames that are stacked')
    parser.add_argument('--max_num_neighbors', type=int, default=2, help='max number of neighbors that we consider')

    config = parser.parse_args()
    visualize_dataset(config)
