# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score


# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
#from plot3D import *

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layer1 = self._conv_layer_set(config.frame_stack, 32)
        self.conv_layer2 = self._conv_layer_set(32, 1)
        self.relu = nn.ReLU()
    
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.ReLU(),
            #nn.AvgPool3d(kernel_size=(3,3,3), stride=(2,2,2))
        )
        return conv_layer

    def forward(self, x, ego_speed=None, ego_command=None):
        #x = self.conv_layer1(x)
        #x = self.conv_layer2(x)
        x = torch.mean(x, dim=(1,2),keepdim=True)
        return x

class Compare(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rep_mix1 = self._2d_conv_layer_set(2,32)
        self.rep_mix2 = self._2d_conv_layer_set(32,1)
    def _2d_conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
        )
        return conv_layer

    def forward(self, ego_rep, other_rep):
        rep = torch.cat((ego_rep, other_rep), dim=1)
        rep = self.rep_mix1(rep)
        rep = self.rep_mix2(rep)
        return rep

class Aggregation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rep_mix1 = self._2d_conv_layer_set(2,32)
        self.rep_mix2 = self._2d_conv_layer_set(32,1)
        self.dim_meta = 1
        self.fc1_meta = nn.Linear(self.dim_meta, 16)
        self.fc2_meta = nn.Linear(16,8)

        #self.fc1 = nn.Linear(65*65+8,256)
        #self.fc1 = nn.Linear(276*276+8, 256)
        self.fc1 = nn.Linear(69*69+8,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,3)
        self.relu = nn.ReLU()

    def _2d_conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.ReLU(),
        )
        return conv_layer

    def forward(self,ego_rep, other_rep, ego_speed, ego_command, num_valid_neighbors=1):
        #other_rep = torch.sum(other_rep, dim=1, keepdim=True)
        #num_valid_neighbors = num_valid_neighbors.view(other_rep.shape[0],1,1,1) 
        #other_rep = torch.div(other_rep, num_valid_neighbors)
        
        #other_rep = torch.mean(other_rep, dim=1, keepdim=True)
        #rep_show = other_rep.detach().cpu().numpy()[0,0,:,:]; plt.imshow(rep_show); plt.show()
        #rep_show = ego_rep.detach().cpu().numpy()[0,0,:,:]; plt.imshow(rep_show); plt.show()
        
        other_rep = torch.mean(other_rep, dim=1, keepdim=True)

        rep = torch.cat((ego_rep, other_rep), dim=1)
        #rep_show = torch.mean(rep,dim=1,keepdim=True).detach().cpu().numpy()[0,0,:,:]; plt.imshow(rep_show); plt.show()
        rep = self.rep_mix1(rep)
        rep = self.rep_mix2(rep)

        #from IPython import embed; embed()
        ego_meta = self.fc1_meta(ego_speed[:,None])
        ego_meta = self.relu(ego_meta)
        ego_meta = self.fc2_meta(ego_meta)
        ego_meta = self.relu(ego_meta)
        
        rep = rep.view(rep.size(0),-1)
        
        rep = torch.cat((rep,ego_meta), dim=-1)
        rep = self.fc1(rep)
        rep = self.relu(rep)
        rep = self.fc2(rep)
        rep = self.relu(rep)
        pred_control = self.fc3(rep)
        return pred_control


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.aggregation = Aggregation(config)
        self.compare = Compare(config)
        self.frame_stack = config.frame_stack
    def forward(self, 
                ego_bev,
                ego_speed, 
                ego_command, 
                other_bev, 
                other_speed, 
                other_transform, # quantized relative transformation
                ego_transform, #ego_transform is inverse transform
                num_valid_neighbors
                ):
        ego_rep = self.encoder(ego_bev, ego_speed, ego_command)
        ego_rep = ego_rep.squeeze(dim=1)

        other_bevs = torch.split(other_bev, self.frame_stack, dim=1)
        other_transforms = torch.split(other_transform, 1, dim=1)
        other_speeds = torch.split(other_speed, 1, dim=1)
        rep=None
        for i in range(len(other_bevs)):
            obs = other_bevs[i]
            spd = other_speeds[i]
            trans = torch.squeeze(other_transforms[i], dim=1)
            other_rep = self.encoder(obs, spd[:,0], ego_command)
            grid = F.affine_grid(trans, size=other_rep.shape)
            other_rep = F.grid_sample(other_rep, grid, mode='bilinear')
            other_rep = other_rep.squeeze(dim=1)
            #rep_show = torch.mean(other_rep,dim=1,keepdim=True).detach().cpu().numpy()[0,0,:,:]; plt.imshow(rep_show); plt.show()
            #rep_show = torch.mean(ego_rep,dim=1,keepdim=True).detach().cpu().numpy()[0,0,:,:]; plt.imshow(rep_show); plt.show()

            #other_rep = self.compare(ego_rep, other_rep)
            if rep is not None:
                rep = torch.cat((rep, other_rep), dim=1)
            else:
                rep = other_rep
        #from IPython import embed; embed()
        #rep_show = torch.mean(rep,dim=1,keepdim=True).detach().cpu().numpy()[0,0,:,:]; plt.imshow(rep_show); plt.show()
        pred_control = self.aggregation(ego_rep, rep, ego_speed, ego_command, num_valid_neighbors)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]
        return pred_throttle, pred_brake, pred_steer


