import time
import argparse

from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tqdm

import glob
import os
import sys
sys.path.append('../')
sys.path.append('../../')
from models.multi_input_model import MultiInputModel
import utils.bz_utils as bzu
from utils.data_loader import get_data_loader as load_data
import pygame
from pygame_display_dataset import PygameDisplayData
from AutoCastSim.AVR import Utils
pygame.init()
def load_model(name, num_output=3):
    if name == 'Naive':
        from models.naive_model import NaiveModel
        net = NaiveModel(num_output)
        return net
    else:
        net = MultiInputModel(num_output)
    return net


def visualize_dataset(config, args):
    data = load_data(args)
    criterion = LocationLoss(choice='l2')
    net = load_model(args.model_name).to(config['device'])
    net.eval()
    model_path = os.path.join(args.log_dir, args.model)
    print("loading model: %s"%model_path)
    net.load_state_dict(torch.load(model_path))
    iterator_tqdm = tqdm.tqdm(data, desc='Validation', total=len(data))
    iterator = enumerate(iterator_tqdm)
    display = PygameDisplayData()
    val_loss = []
    for i, (birdview, ego_meta, target_location, lidar, ego_transform) in iterator:
        birdview = birdview.to(config['device'], dtype=torch.float)
        ego_meta = ego_meta.to(config['device'], dtype=torch.float)
        target_location = target_location.to(config['device'], dtype=torch.float)
        if args.model_name =='Naive':
            pred_location = net(birdview)
        else:
            pred_location = net(birdview, ego_meta)
        loss = criterion(pred_location, target_location)
        loss_mean = loss.mean()
        val_loss.append(loss_mean.item())
        print("Loss:", loss_mean.item(), 
              " Predicted Location: ", pred_location.cpu().detach().numpy()[0],
              " Target Location: ", target_location.cpu().detach().numpy()[0])

        display.render( lidar[0], 
                        pred_location.cpu().detach().numpy(), 
                        target_location.cpu().detach().numpy(),
                        ego_transform)
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
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-dataloader-workers', type=int, default=0, help='number of pytorch dataloader workers. set it to zero for debugging')
    parser.add_argument('--model', type=str, default='model-128.th', help='path to the model to be loaded')
    parser.add_argument('--model_name', type=str, default='MultiInput', help='specific model to be loaded')
    parser.add_argument('--log_dir', type=str, default='../../training', help='dir that stores model')
    parser.add_argument('--shared', type=bool, default=False, help='whether use shared lidar data')
    args = parser.parse_args()
    config = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'data_args': {
                'dataset_dir': args.dataset_dir,
                },
            'model_args': {
                },
            }

    visualize_dataset(config, args)
