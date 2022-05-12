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

import itertools

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layer1 = self._conv_layer_set(config.frame_stack * 10, 128, stride = 2)
        self.conv_layer2 = self._conv_layer_set(128, 64, stride = 1)
        self.conv_layer3 = self._conv_layer_set(64, 4, stride = 2)
        self.relu = nn.ReLU()
    
    def _conv_layer_set(self, in_c, out_c, stride):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=stride, padding=0),
            nn.ReLU(),
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        return x

class V2VNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.encoder_other = Encoder(config)
        self.relu = nn.ReLU()
        self.conv1 = GCNConv(68*68*4, config.num_hidden, aggr='max')
        self.conv2 = GCNConv(config.num_hidden, config.num_hidden, aggr='max')
        self.conv3 = GCNConv(config.num_hidden, config.num_hidden, aggr='max')
        self.fc1 = nn.Linear(config.num_hidden+8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dim_meta = 1
        self.fc1_meta = nn.Linear(self.dim_meta, 16)
        self.fc2_meta = nn.Linear(16, 8)
        self.device = config.device
        self.max_num_neighbors = config.max_num_neighbors

    def forward(self, ego_lidar, ego_speed, other_lidar, other_transform):
        
        other_lidars = torch.split(other_lidar, 1, dim=1)
        other_transforms = torch.split(other_transform, 1, dim=1)

        #other_bev1, other_bev2 = torch.squeeze(other_bev1, dim = 1), torch.squeeze(other_bev2, dim = 1)
        # Encode representations
        ego_rep = self.encoder(ego_lidar)
        ego_rep_show = ego_lidar
        ego_rep_show_shape = ego_lidar.shape
        ego_rep_show = np.max(ego_rep_show[0].detach().cpu().numpy(),axis=0)
        ego_shape = ego_rep.shape
        ego_rep = ego_rep.contiguous().view(ego_rep.size(0),-1)
        x = ego_rep
        other_show = []
        for i in range(len(other_lidars)):
            lidar = torch.squeeze(other_lidars[i], dim=1)
            other_rep = self.encoder_other(lidar)
            other_rep_show = lidar
            # Load Transformations (N, 3, 4)
            indice = torch.tensor([0,1,3])
            trans = torch.squeeze(other_transforms[i], dim=1)[:,:2,indice]
            grid = F.affine_grid(trans, size = ego_shape)
            other_rep_ego_frame = F.grid_sample(other_rep, grid, mode='bilinear')
            
            grid_show = F.affine_grid(trans, size = ego_rep_show_shape)
            other_rep_show_ego_frame = F.grid_sample(other_rep_show, grid_show, mode='bilinear')
            other_show.append(np.max(other_rep_show_ego_frame[0].detach().cpu().numpy(), axis=0))
            other_rep_ego_frame = other_rep_ego_frame.contiguous().view(other_rep_ego_frame.size(0),-1)
            x = torch.cat([x, other_rep_ego_frame], axis=0) 
        from matplotlib import pyplot as plt
        #if int(frame_id)>=100:
        #    from IPython import embed; embed() 
        # Specify Edge Index
        index_template = torch.tensor([[0, 0, 0, 1, 1, 2, 1, 2, 3, 2, 3, 3],
                                       [1, 2, 3, 2, 3, 3, 0, 0, 0, 1, 1, 2]], dtype=torch.long)
        nodes = np.arange(self.max_num_neighbors)
        edges = []
        for z in itertools.permutations(nodes,2):
            edges.append(z)
        edges = np.array(edges)
        edges = edges.T
        index_template = torch.tensor(edges, dtype=torch.long)
        batch_size = ego_lidar.shape[0]
        index_template = batch_size * index_template
        edge_index = index_template
        for i in range(batch_size-1):
            edge_index = torch.cat([edge_index,index_template + i + 1], dim=1)
        edge_index = edge_index.to(self.device)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        #[ Node 0 x N, Node 1 x N, Node 2 x N, Node 3 x N]
        batch = np.arange(batch_size)
        batch = np.tile(batch,(1,4))
        batch = torch.from_numpy(batch).squeeze(dim=0).to(self.device)
        global_rep = global_max_pool(x, batch) # [batch_size, hidden channels]
        
        #from IPython import embed; embed()
        ego_meta = self.fc1_meta(ego_speed[:,None])
        ego_meta = self.relu(ego_meta)
        ego_meta = self.fc2_meta(ego_meta)
        ego_meta = self.relu(ego_meta)
        global_rep = torch.cat((global_rep, ego_meta), dim=-1)
        
        global_rep = self.fc1(global_rep)
        global_rep = self.relu(global_rep)
        global_rep = self.fc2(global_rep)
        global_rep = self.relu(global_rep)
        
        pred_control = self.fc3(global_rep)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]
        return pred_throttle, pred_brake, pred_steer


class VoxelNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4624*4+8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dim_meta = 1
        self.fc1_meta = nn.Linear(self.dim_meta, 16)
        self.fc2_meta = nn.Linear(16, 8)
        self.device = config.device

    def forward(self, ego_lidar, ego_speed, other_lidar=None, other_transform=None):
        
        # Encode representations
        ego_rep = self.encoder(ego_lidar)
        ego_shape = ego_rep.shape
        ego_rep = ego_rep.contiguous().view(ego_rep.size(0),-1)
        x = ego_rep
        
        ego_meta = self.fc1_meta(ego_speed[:,None])
        ego_meta = self.relu(ego_meta)
        ego_meta = self.fc2_meta(ego_meta)
        ego_meta = self.relu(ego_meta)
        global_rep = torch.cat((x, ego_meta), dim=-1)
        #from IPython import embed; embed() 
        global_rep = self.fc1(global_rep)
        global_rep = self.relu(global_rep)
        global_rep = self.fc2(global_rep)
        global_rep = self.relu(global_rep)
        
        pred_control = self.fc3(global_rep)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]
        return pred_throttle, pred_brake, pred_steer

