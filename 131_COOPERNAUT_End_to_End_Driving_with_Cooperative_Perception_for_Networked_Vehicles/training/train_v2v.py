import argparse
import tqdm
import wandb
import torch

import numpy as np
import torch.nn.functional as F
from torch import optim
from copy import deepcopy
from models import V2VNet, VoxelNet
from .logger import Logger
from utils.voxel_transformer_loader import get_data_loader

def train(config):
    # Config models, optimzers, loss functions, datasets and logger
    if config.max_num_neighbors>0 and not config.earlyfusion:
        model = V2VNet(config).to(config.device)
    else:
        model = VoxelNet(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.init_lr, weight_decay=1e-5)
    dataset = get_data_loader(config)
    logger = Logger(config)
    if config.finetune is not None:
        print("Loading Finetune Model:", config.finetune)
        model.load_state_dict(torch.load(config.finetune))
    if config.eval_data is not None:
        print("Loading Validation Dataset:", config.eval_data)
        eval_config = deepcopy(config)
        eval_config.data = config.eval_data
        #eval_config.batch_size = 1
    wandb.watch(model,log_freq=100) 
    # Start training
    model.train()
    global_steps = 0
    moving_loss = []
    val_loss = 0
    for epoch in tqdm.tqdm(range(config.num_epochs)):
        for batch_data in dataset:

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
            other_lidar = other_lidar.float().to(config.device)
            other_transform = other_transform.float().to(config.device)
            num_valid_neighbors = num_valid_neighbors.float().to(config.device)
            pred_throttle, pred_brake, pred_steer = model(ego_lidar,ego_speed, other_lidar, other_transform)
            coeff_throttle = 1.0
            coeff_brake = 1.0
            coeff_steer = 1.0
            throttle_loss = F.l1_loss(pred_throttle, ego_throttle)
            brake_loss = F.l1_loss(pred_brake, ego_brake)
            steer_loss = F.l1_loss(pred_steer, ego_steer)
            loss = coeff_throttle * throttle_loss  + coeff_brake * brake_loss + coeff_steer * steer_loss
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            moving_loss.append(_numpy(loss))

            if global_steps % config.num_steps_per_log == 0:
                if global_steps % (50*config.num_steps_per_log) == 0:
                    if config.eval_data is not None:
                        print("evaluating")
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                            val_loss = evaluate(eval_config,model)
                        print("val loss:", val_loss)
                        model.train()

                logger.log(
                    global_steps, 
                    _numpy(bev_rgb[0]), 
                    float(np.array(moving_loss).mean()),
                    val_loss,
                    None,None,None,None
                )
                moving_loss = []
            global_steps += 1
            print("Step ", global_steps," Loss:", loss.item(), "Throttle Loss:", throttle_loss.item(), "Brake Loss:", brake_loss.item(), "Steer Loss:", steer_loss.item())
            if epoch==0: break
        if epoch % config.checkpoint_frequency == 0 or (epoch == config.num_epochs - 1):
            logger.save(epoch,model)

def _numpy(x):
    return x.detach().cpu().numpy()

def evaluate(config, model):
    model.eval()
    torch.cuda.empty_cache()
    dataset = get_data_loader(config)
    moving_loss = []
    for batch_data in dataset:
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
        other_lidar = other_lidar.float().to(config.device)
        other_transform = other_transform.float().to(config.device)
        num_valid_neighbors = num_valid_neighbors.float().to(config.device)
        pred_throttle, pred_brake, pred_steer = model(ego_lidar,ego_speed, other_lidar, other_transform)
        coeff_throttle = 1.0
        coeff_brake = 1.0
        coeff_steer = 1.0
        throttle_loss = F.l1_loss(pred_throttle, ego_throttle)
        brake_loss = F.l1_loss(pred_brake, ego_brake)
        steer_loss = F.l1_loss(pred_steer, ego_steer)
        loss = coeff_throttle * throttle_loss  + coeff_brake * brake_loss + coeff_steer * steer_loss
            
        moving_loss.append(_numpy(loss))

    return float(np.array(moving_loss).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataloader configs
    parser.add_argument('--data', required=True, help='path to the dataset. it should contains folders named 0, 1, etc. check: https://drive.google.com/drive/u/1/folders/1xmZiu9yiFw2IdQETKtA4KyiXPWh-2MIH')
    parser.add_argument('--daggerdata', default=None,
                        help='path to the dagger dataset. it should contains folders named 0, 1, etc.')
    parser.add_argument('--eval-data', default=None, help='path to the validation dataset')
    parser.add_argument('--ego-only', action='store_true', help='only return data of one agent', default=True)
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-dataloader-workers', type=int, default=8, help='number of pytorch dataloader workers. set it to zero for debugging')
    parser.add_argument('--use-lidar', action='store_true', help='Lidar information as the 4th data returned')
    parser.add_argument('--visualize', action='store_true', help='wether we are visualizing')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--num-commands', type=int, default=6)
    parser.add_argument('--frame-stack', type=int, default=1, help='num of frames that are stacked') 
    parser.add_argument('--max_num_neighbors', type=int, default=3, help='max number of neighbors that we consider')

    # Model configs
    parser.add_argument('--num-hidden', type=int, default=32)
    parser.add_argument('--num-node-features', type=int, default=10)
    
    # Traninig configs
    parser.add_argument('--project', default='autocast')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--init-lr', type=float, default=3e-4)
    parser.add_argument('--num-steps-per-log', type=int, default=10)
    parser.add_argument('--checkpoint-frequency', type=int, default=25)
    parser.add_argument('--finetune',type=str, default=None)

    # V2V Model
    parser.add_argument('--v2v', action='store_true')
    # Early Fusion Model
    parser.add_argument('--earlyfusion', action='store_true')
    config = parser.parse_args()

    train(config)
