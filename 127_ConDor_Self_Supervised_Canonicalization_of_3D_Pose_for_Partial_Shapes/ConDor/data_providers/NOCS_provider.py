import numpy as np
import tensorflow as tf
import os, sys
sys.path.append("../")
from utils.data_prep_utils import load_h5_normals_files, load_h5_files
from utils.pointclouds_utils import pc_batch_preprocess, pc_normals_batch_preprocess
from pooling import kdtree_indexing_
from scipy.spatial import cKDTree
from multiprocessing import Pool
import h5py
import time
from data_providers import classifiaction_provider as cf_provider


class NOCSProvider(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_path, batch_size=8, shuffle=True):
        'Initialization'
        
        
        self.full, self.partial, self.idx_partial_to_full = self.load_h5_files(file_path)
        
        
        self.n_points_full = self.full.shape[1]
        self.n_points_partial = self.partial.shape[1]
        self.n_dims = self.full.shape[2]
        self.n_samples = self.full.shape[0]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = list()
        
        self.on_epoch_end()
        self.i_ = 0
        self.time0 = time.time()
        self.preprocess_time = 0

    def load_h5_files(self, data_path):
        '''
        Load the h5 dataset
        '''
        
        dataset_file = h5py.File(data_path, "r")
        
        full = np.asarray(dataset_file["full"])
        partial = np.asarray(dataset_file["partial"])
        idx_partial_to_full = np.asarray(dataset_file["idx"])
        
        return full, partial, idx_partial_to_full

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, X_partial, idx_partial_to_full = self.__data_generation(indexes)
        return {"full": tf.cast(X, tf.float32), "partial": tf.cast(X_partial, tf.float32), "idx_partial_to_full": idx_partial_to_full}

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.i_ = 0

    def __data_generation(self, indexes):

        preproc_time = time.time()

        idx_partial_shuffle = np.random.permutation(np.arange(self.n_points_partial))
        idx_full_remaining = np.random.permutation(np.arange(self.n_points_partial, self.n_points_full))

        X = self.full[indexes, ...]
        X_partial = self.partial[indexes, ...]
        X_idx_partial_to_full = self.idx_partial_to_full[indexes, ...]
        
        X_partial_shuffle = X_partial[:, idx_partial_shuffle, :]
        
        
        X_idx_partial_to_full = X_idx_partial_to_full[:, idx_partial_shuffle] 
        X_full_shuffle_remaining = X[:, idx_full_remaining, :]
        
        X_full_shuffle_partial = tf.gather(X, X_idx_partial_to_full, batch_dims = 1)
        X_shuffle = np.hstack([X_full_shuffle_partial, X_full_shuffle_remaining])     
        X_idx = np.stack([np.arange(self.n_points_partial)]*self.batch_size, axis = 0)
           
        self.preprocess_time += (time.time()-preproc_time)
        self.i_ += 1
        
        return X_shuffle, X_partial_shuffle, X_idx

    def set_preprocessing(self, preprocessing):
        self.preprocess = preprocessing

    def get_data(self):
        return self.full, self.partial, self.idx_partial_to_full


class DRACOProvider(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, file_path, batch_size=8, shuffle=True):
        'Initialization'

        self.full, self.partial, self.idx_partial_to_full, self.depth, self.pose = self.load_h5_files(
            file_path)

        self.n_points_full = self.full.shape[1]
        self.n_points_partial = self.partial.shape[1]
        self.n_dims = self.full.shape[2]
        self.n_samples = self.full.shape[0]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = list()

        self.on_epoch_end()
        self.i_ = 0
        self.time0 = time.time()
        self.preprocess_time = 0

    def load_h5_files(self, data_path):
        '''
        Load the h5 dataset
        '''

        dataset_file = h5py.File(data_path, "r")

        full = np.asarray(dataset_file["full"])
        partial = np.asarray(dataset_file["partial"])
        idx_partial_to_full = np.asarray(dataset_file["idx"])
        depth = np.asarray(dataset_file["depth"])
        pose = np.asarray(dataset_file["pose"])

        return full, partial, idx_partial_to_full, depth, pose

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, X_partial, idx_partial_to_full, X_depth, X_pose = self.__data_generation(indexes)
        return {"full": tf.cast(X, tf.float32), "partial": tf.cast(X_partial, tf.float32), "idx_partial_to_full": idx_partial_to_full, "depth": tf.cast(X_depth, tf.float32), "pose": tf.cast(X_pose, tf.float32)}

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.i_ = 0

    def __data_generation(self, indexes):

        preproc_time = time.time()

        idx_partial_shuffle = np.random.permutation(
            np.arange(self.n_points_partial))
        idx_full_remaining = np.random.permutation(
            np.arange(self.n_points_partial, self.n_points_full))

        X = self.full[indexes, ...]
        X_partial = self.partial[indexes, ...]
        X_depth = self.depth[indexes, ...]
        X_pose = self.pose[indexes, ...]
        X_idx_partial_to_full = self.idx_partial_to_full[indexes, ...]

        X_partial_shuffle = X_partial[:, idx_partial_shuffle, :]
        X_depth_shuffle = X_depth[:, idx_partial_shuffle, :]
        X_idx_partial_to_full = X_idx_partial_to_full[:, idx_partial_shuffle]
        X_full_shuffle_remaining = X[:, idx_full_remaining, :]

        X_full_shuffle_partial = tf.gather(
            X, X_idx_partial_to_full, batch_dims=1)
        X_shuffle = np.hstack(
            [X_full_shuffle_partial, X_full_shuffle_remaining])
        X_idx = np.stack([np.arange(self.n_points_partial)]
                         * self.batch_size, axis=0)

        self.preprocess_time += (time.time()-preproc_time)
        self.i_ += 1

        return X_shuffle, X_partial_shuffle, X_idx, X_depth_shuffle, X_pose

    def set_preprocessing(self, preprocessing):
        self.preprocess = preprocessing

    def get_data(self):
        return self.full, self.partial, self.idx_partial_to_full


if __name__ == "__main__":


    dataset_path = "/home/husky/Documents/EquiNet/NOCS-dataset-generator/src/cars.h5"
    dataset_nocs_provider = NOCSProvider(dataset_path) 
    
    for i, data in enumerate(dataset_nocs_provider):
        

        full_pcd = data["full"]
        partial_pcd = data["partial"]
        idx_partial_to_full = data["idx_partial_to_full"]


        # idx_inv takes from kdtree indices to original
        # idx takes from original to kdtree
        full_pcd_kd, idx, idx_inv = kdtree_indexing_(full_pcd)
        partial_pcd_kd, idx_partial, idx_inv_partial = kdtree_indexing_(partial_pcd)
                

        out = tf.gather_nd(full_pcd_kd, idx_inv) # kd to input full
        out = tf.gather(out, idx_partial_to_full, batch_dims = 1) # input full to input partial
        out = tf.gather_nd(out, idx_partial) # input partial to kdtree
        
        print(out - partial_pcd_kd)
        
        for key in data.keys():
            print(key, " ", data[key].shape)
        print("\n")


    pass
