from __future__  import print_function
import os
import json
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import random, sys

# from utils.pointcloud_utils import pc_batch_preprocess
import h5py


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class H5Loader(Dataset):
    
    def __init__(self, data_path, files_list = [], n_points = 1024,
                 preprocess=list()):
        '''
        Load DFAUST dataset
        '''
        
        files_list_full = []
        print(data_path)
        for i in range(len(files_list)):
            files_list_full.append(os.path.join(data_path,"") +  files_list[i])
        
        files_list = files_list_full.copy()
        print(files_list)

        data = []
        for file in files_list_full:
            print(file)
            f = h5py.File(file, "r")
            print(f.keys())
            data.append(f['data'][:])


        self.data = np.concatenate(data, axis=0)
        self.n_points_data = np.shape(self.data)[1]
        self.n_samples = self.data.shape[0]
        self.nv = self.data.shape[1]
        self.n_points = n_points
        self.preprocess = preprocess
        print("Data processed with number of samples = ", self.data.shape)

    def __len__(self):
        '''
        Total number of data points
        '''
    
        return self.data.shape[0]

    def __getitem__(self, index):
        '''
        Get item for Pytorch dataset
        "pc", "labels"
        '''

        data = self.__data_generation(index)
        
        return data

    def __data_generation(self, index):

        idx = np.random.permutation(np.arange(self.n_points_data))[:self.n_points]

        X = self.data[index, ...]
        X = X[idx, ...]


        X = np.expand_dims(X, 0)

        
        # for i in range(len(self.preprocess)):
        #     X, y = pc_batch_preprocess(X, y=None, proc=self.preprocess[i])

        X = X.squeeze()

        data = {"pc": torch.Tensor(X)}

        return data

    def set_preprocessing(self, preprocessing):
        self.preprocess = preprocessing

    def get_data(self):
        return self.data

    def get_num_points(self):
        return self.n_points

    def get_preprocess(self):
        return self.preprocess


    
if __name__=="__main__":

    # dataset_path = "../../../data/shapenet_segmentation/hdf5_data/"
    # dataset_path = "/home/rahul/Internship/Brown2022/ArticulatedNOCS/CanonicalArticulations/dataset/dfaust/"
    dataset_path = "/home/rahul/Internship/Brown2021/code/data/shapenet_single_class/data_hdf5"
 
    dataset_files = ["test_aero.h5"]
    n_points = 1024
    batch_size=2
    preprocess = []#["rotate_z"]
    shuffle=True

    data_set = H5Loader(data_path = dataset_path, files_list = dataset_files, n_points = n_points, 
    preprocess = preprocess)

    loader = DataLoader(dataset=data_set, batch_size = 8, shuffle=True)
    for l, batch in enumerate(loader):
        
        
        print(l, batch["pc"].shape)
