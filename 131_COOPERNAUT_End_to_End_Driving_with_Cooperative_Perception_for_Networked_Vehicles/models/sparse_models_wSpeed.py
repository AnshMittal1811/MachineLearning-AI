import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .sparse_resnet import ResNet14, ResNet18, ResNet34, ResNet50, ResNet101


class SparsePolicyNetSpeed(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Configs
        self.T = config.T
        self.num_commands = config.num_commands
        
        self.backbone = nn.Sequential(
            ResNet14(1,config.num_hidden,D=2),
            ME.MinkowskiGlobalPooling(),
        )
        
        self.spd_encoder = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(True),
            nn.Linear(16,16),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.num_hidden+16,128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128,(config.T*2+1)*config.num_commands)            
        )
        
    def forward(self, bev, ego_speed, command):
        
        hidden = self.backbone(bev)
        hidden = torch.cat([hidden.F, self.spd_encoder(ego_speed[:,None])], dim=-1)

        outputs = self.fc(hidden)
        pred_locs = outputs[:,:-self.num_commands].view(-1,self.num_commands,self.T,2)
        pred_brakes = outputs[:,-self.num_commands:]
        
        pred_loc = pred_locs.gather(1,command[:,None,None,None].repeat(1,1,self.T,2)).squeeze(1)
        pred_brake = pred_brakes.gather(1,command[:,None]).squeeze(1)


        return pred_loc, pred_brake
