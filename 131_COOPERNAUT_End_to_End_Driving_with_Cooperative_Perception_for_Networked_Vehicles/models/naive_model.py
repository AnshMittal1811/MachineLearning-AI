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

# Create Naive Conv3d Model
class NaiveModel(nn.Module):
    def __init__(self, num_output):
        super(NaiveModel, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 16)
        self.fc1 = nn.Linear(864, 128)
        self.fc2 = nn.Linear(128, num_output)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128) #Determine if batch is needed
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((1, 3, 3)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.batch(x)
        #x = self.drop(x)
        x = self.fc2(x)
        
        return x
