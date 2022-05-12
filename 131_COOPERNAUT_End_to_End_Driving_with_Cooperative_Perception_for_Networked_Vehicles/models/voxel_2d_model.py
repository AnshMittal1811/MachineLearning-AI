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

class VoxelModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 16)
        self.fc1 = nn.Linear(864, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128) #Determine if batch is needed
        self.drop=nn.Dropout(p=0.15)        
                
        dim_concat = 64
        self.final_fc1 = nn.Linear(dim_concat, 64)
        self.final_fc2 = nn.Linear(64, 3)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 3, 3)),
        )
        return conv_layer
    
    def forward(self, x, ego_speed=None, ego_command=None):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        concat_output = x
        concat_output = self.final_fc1(concat_output)
        concat_output = self.relu(concat_output)
        pred_control = self.final_fc2(concat_output)
        
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]
        return pred_throttle, pred_brake, pred_steer

class VoxelMetaModel(VoxelModel):
    def __init__(self, config):
        super().__init__(config)
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 16)
        self.fc1 = nn.Linear(864, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.LeakyReLU()
        
        dim_meta = 1
        self.fc1_meta = nn.Linear(dim_meta, 16)
        self.fc2_meta = nn.Linear(16, 8) 
        
        dim_concat = 64+8
        self.final_fc1 = nn.Linear(dim_concat, 64)
        self.final_fc2 = nn.Linear(64, 3)

    def forward(self, x, ego_speed=None, ego_command=None):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        ego_meta = self.fc1_meta(ego_speed[:,None])
        ego_meta = self.relu(ego_meta)
        ego_meta = self.fc2_meta(ego_meta)
        concat_output = torch.cat((x, ego_meta), dim=1)
        concat_output = self.final_fc1(concat_output)
        concat_output = self.relu(concat_output)
        pred_control = self.final_fc2(concat_output)
        
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]
        return pred_throttle, pred_brake, pred_steer


