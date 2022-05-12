import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch import optim
from models import SparsePolicyNet, SparsePolicyNetSpeed, SparseControlNet, SparseSpeedControlNet, SpeedOnlyControlNet
import numpy as np
from .logger import Logger

def train(config):
    learn_control=False
    classification=False
    # Config models, optimzers, loss functions, datasets and logger
    if config.use_speed:
        model = SparseSpeedControlNet(config).to(config.device)
        #model = SpeedOnlyControlNet(config).to(config.device)
    else:
        model = SparseControlNet(config).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.init_lr, weight_decay=1e-5)
    
    if config.frame_stack <=1:
        from utils.data_loader import get_data_loader
    else:
        print("Using multiframe data loader")
        from utils.multiframe_data_loader import get_data_loader
    dataset = get_data_loader(config)
    logger = Logger(config)
    if config.finetune is not None:
        print("Loading:", config.finetune)
        model.load_state_dict(torch.load(config.finetune))
    model.train()
    # Start training
    global_steps = 0
    moving_loss = []
    for epoch in tqdm.tqdm(range(config.num_epochs)):
        for batch_data in dataset:

            bev_rgb, bev_coord, bev_feat, ego_future, ego_speed, ego_brake, ego_has_plan, ego_command, ego_control = batch_data

            # Convert tensors
            bev = ME.SparseTensor(bev_feat.float(), bev_coord).to(config.device)
            ego_future = ego_future.float().to(config.device)
            ego_speed = ego_speed.float().to(config.device)
            #ego_speed = ME.SparseTensor(ego_speed.float()).to(config.device)
            ego_brake = ego_brake.float().to(config.device)
            ego_throttle = ego_control[:,0].float().to(config.device)
            ego_steer = ego_control[:,2].float().to(config.device)
            ego_has_plan = ego_has_plan.byte().to(config.device)
            ego_command = ego_command.long().to(config.device)
            ego_control = ego_control.float().to(config.device)
            
            pred_throttle, pred_brake, pred_steer = model(bev, ego_speed, ego_command)
            #loss = F.mse_loss(pred_control, ego_control.float())
            coeff_throttle = 0.5
            coeff_brake = 1.0
            coeff_steer = 1.0
            throttle_loss = F.mse_loss(pred_throttle, ego_throttle)
            brake_loss = F.mse_loss(pred_brake, ego_brake)
            steer_loss = F.mse_loss(pred_steer, ego_steer)
            loss = coeff_throttle * throttle_loss  + coeff_brake * brake_loss + coeff_steer * steer_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            moving_loss.append(_numpy(loss))

            global_steps += 1
            print("Step ", global_steps," Loss:", loss, "Throttle Loss:", throttle_loss, "Brake Loss:", brake_loss, "Steer Loss:", steer_loss)#, " future_loss:", future_loss, "brake_loss:", brake_loss)
            '''
            if global_steps % config.num_steps_per_log == 0:

                logger.log(
                    global_steps, 
                    _numpy(bev_rgb[0]), 
                    float(future_loss.mean()),
                    _numpy(pred_ego_future[0]), _numpy(ego_future[0]),
                    _numpy(pred_brake[0]), _numpy(ego_brake[0])
                )
            '''
        logger.log(
                    global_steps, 
                    _numpy(bev_rgb[0]), 
                    float(np.array(moving_loss).mean()),
                    None,None,None,None
                )
        if epoch % config.checkpoint_frequency == 0 or (epoch == config.num_epochs - 1):
            logger.save(
                    epoch,
                    model
                    )
        moving_loss = []



def _numpy(x):
    return x.detach().cpu().numpy()


if __name__ == '__main__':
    
    import argparse
    import tqdm
    parser = argparse.ArgumentParser()

    # Dataloader configs
    parser.add_argument('--data', required=True, help='path to the dataset. it should contains folders named 0, 1, etc. check: https://drive.google.com/drive/u/1/folders/1xmZiu9yiFw2IdQETKtA4KyiXPWh-2MIH')
    parser.add_argument('--ego-only', action='store_true', help='only return data of one agent', default=True)
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-dataloader-workers', type=int, default=8, help='number of pytorch dataloader workers. set it to zero for debugging')
    parser.add_argument('--use-lidar', action='store_true', help='Lidar information as the 4th data returned')
    parser.add_argument('--visualize', action='store_true', help='wether we are visualizing')
    parser.add_argument('--use-speed', action='store_true', help='wether to use speed in the model')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('-T', type=int, default=10, help='num of waypoints to predict')
    parser.add_argument('--num-commands', type=int, default=6)
    parser.add_argument('--frame-stack', type=int, default=1, help='num of frames that are stacked') 
    # Model configs
    parser.add_argument('--num-hidden', type=int, default=256)
    
    
    # Traninig configs
    parser.add_argument('--project', default='autocast')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--init-lr', type=float, default=3e-4)
    parser.add_argument('--num-steps-per-log', type=int, default=10)
    parser.add_argument('--checkpoint-frequency', type=int, default=25)
    parser.add_argument('--finetune',type=str, default=None)


    config = parser.parse_args()

    train(config)
