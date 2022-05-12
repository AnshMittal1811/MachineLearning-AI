import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .sparse_resnet import ResNet14, ResNet18, ResNet34, ResNet50, ResNet101


class SparsePolicyNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Configs
        self.T = config.T
        self.num_commands = config.num_commands
        self.frame_stack = config.frame_stack
        print("Stacking frame :", self.frame_stack)
        if self.frame_stack > 1:
            self.backbone = nn.Sequential(
                ResNet50(1,config.num_hidden,D=3),
                ME.MinkowskiGlobalPooling(),
            )
        else:
            self.backbone = nn.Sequential(
                ResNet50(1,config.num_hidden,D=2),
                ME.MinkowskiGlobalPooling(),
            )
        
        self.spd_encoder = nn.Sequential(
            nn.Linear(1,1),
            nn.ReLU(True),
            nn.Linear(1,1),
        )
        
        self.fc = nn.Sequential(
            #nn.Linear(config.num_hidden+64,128),
            nn.Linear(config.num_hidden,128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128,(config.T*2+1)*config.num_commands)            
        )
        
    def forward(self, bev, ego_speed, command):
        
        hidden = self.backbone(bev)
        #hidden = torch.cat([hidden.F, self.spd_encoder(ego_speed[:,None])], dim=-1)

        outputs = self.fc(hidden.F)
        pred_locs = outputs[:,:-self.num_commands].view(-1,self.num_commands,self.T,2)
        pred_brakes = outputs[:,-self.num_commands:]
        
        pred_loc = pred_locs.gather(1,command[:,None,None,None].repeat(1,1,self.T,2)).squeeze(1)
        pred_brake = pred_brakes.gather(1,command[:,None]).squeeze(1)

        return pred_loc, pred_brake

class SparseControlNet(SparsePolicyNet):
    def __init__(self, config):
        super().__init__(config)
        self.control_fc = nn.Sequential(
                nn.Linear(config.num_hidden, 128),
                nn.ReLU(True),
                nn.Linear(128,128),
                nn.ReLU(True),
                nn.Linear(128,3) #Throttle, Steer, Brake
                )

    def forward(self, bev, ego_speed, command):
        hidden = self.backbone(bev)
        pred_control = self.control_fc(hidden.F)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]
        return pred_throttle, pred_brake, pred_steer


class SparseSpeedControlNet(SparseControlNet):
    def __init__(self, config):
        super().__init__(config)
        if self.frame_stack > 1:
            self.backbone = nn.Sequential(
                ResNet50(1,config.num_hidden,D=3),
                ME.MinkowskiGlobalPooling(),
            )
        else:
            self.backbone = nn.Sequential(
                ResNet50(1,config.num_hidden,D=2),
                ME.MinkowskiGlobalPooling(),
            )

        self.control_fc = nn.Sequential(
                nn.Linear(config.num_hidden+1, 128),
                nn.ReLU(True),
                nn.Linear(128,128),
                nn.ReLU(True),
                nn.Linear(128,3) #Throttle, Steer, Brake
                )
        self.BN = nn.BatchNorm1d(config.num_hidden)

    def forward(self, bev, ego_speed, command):
        hidden = self.backbone(bev)
        hidden = torch.cat([hidden.F, ego_speed[:,None]], dim=-1)
        pred_control = self.control_fc(hidden)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]
        return pred_throttle, pred_brake, pred_steer

class SpeedOnlyControlNet(SparseControlNet):
    def __init__(self, config):
        super().__init__(config)
        self.spd_control_fc = nn.Sequential(
            nn.Linear(1,128),
            nn.ReLU(True),
            nn.Linear(128,3),
        )

    def forward(self, bev, ego_speed, command):
        hidden = ego_speed[:,None]
        pred_control = self.spd_control_fc(hidden)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]
        return pred_throttle, pred_brake, pred_steer

