import numpy as np
import tensorflow as tf

from utils.data_prep_utils import load_h5_data_files, load_h5_data_files_new
from utils.pointclouds_utils import pc_batch_preprocess, pc_normals_batch_preprocess
from scipy.spatial import cKDTree
from utils.pointclouds_utils import uniform_pc_sampling


from multiprocessing import Pool
import h5py
import os
import time

def kdtree_index(X):
    nb = X.shape[0]
    for i in range(nb):
        x = X[i, ...]
        T = cKDTree(x)
        X[i, ...] = np.take(x, T.indices, axis=0)
    return X

def tile(x, rep=16):
    return np.tile(x, (16, 1))

def load_dataset(dataset, batch_size, num_points, shuffle, train_files_list, val_files_list, test_files_list):
    

    train_files_list = train_files_list
    val_files_list = val_files_list
    test_files_list = test_files_list

    train_data_folder = dataset['train_data_folder']
    val_data_folder = dataset['val_data_folder']
    test_data_folder = dataset['test_data_folder']

    train_preprocessing = dataset['train_preprocessing']
    val_preprocessing = dataset['val_preprocessing']
    test_preprocessing = dataset['test_preprocessing']
    NUM_SAMPLES_DOWNSAMPLED = None

    train_provider = Provider_new(files_list=train_files_list,
                              data_path=train_data_folder,
                              n_points=num_points,
                              n_samples=NUM_SAMPLES_DOWNSAMPLED,
                              batch_size=batch_size,
                              preprocess=train_preprocessing,
                              shuffle=shuffle)

    val_provider = Provider_new(files_list=val_files_list,
                            data_path=val_data_folder,
                            n_points=num_points,
                            n_samples=NUM_SAMPLES_DOWNSAMPLED,
                            batch_size=batch_size,
                            preprocess=val_preprocessing,
                            shuffle=shuffle)

    test_provider = Provider_new(files_list=test_files_list,
                             data_path=test_data_folder,
                             n_points=num_points,
                             n_samples=NUM_SAMPLES_DOWNSAMPLED,
                             batch_size=batch_size,
                             preprocess=test_preprocessing,
                             shuffle=False)

    return train_provider, val_provider, test_provider


class Provider_new(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files_list, data_path, n_points, n_samples=None,
                 batch_size=32, preprocess=list(), sampler=None, shuffle=True):
        'Initialization'

        if sampler is None:
            self.sampler = uniform_pc_sampling
        else:
            self.sampler = sampler
        self.n_samples = n_samples
        self.data = load_h5_data_files_new(data_path, files_list)
        print(self.data.shape)
        self.n_points_data = np.shape(self.data)[1]
        self.n_shapes = self.data.shape[0]
        self.nv = self.data.shape[1]
        self.n_points = n_points

        self.batch_size = batch_size

        self.shuffle = shuffle
        self.preprocess = preprocess
        # self.pool = Pool(4)
        self.on_epoch_end()
        self.i_ = 0
        self.time0 = time.time()
        self.preprocess_time = 0
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_shapes / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(indexes)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_shapes)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.i_ = 0
    def __data_generation(self, indexes):
        preproc_time = time.time()
        idx = np.random.permutation(np.arange(self.n_points_data))[:self.n_points]
        Y = self.data[indexes, ...]
        Y = Y[:, idx, ...]
        for i in range(len(self.preprocess)):
            Y = pc_batch_preprocess(x=Y, f=None, y=None, proc=self.preprocess[i])
        Y = Y.astype(dtype=np.float32)
        self.preprocess_time += (time.time()-preproc_time)
        # print('time_ratio = ', self.preprocess_time / (time.time() - self.time0))
        # print(self.i_*self.batch_size)
        self.i_ += 1
        # X = self.sampler(Y, self.n_samples)
        return Y

    def set_preprocessing(self, preprocessing):
        self.preprocess = preprocessing

    def get_data(self):
        return self.data

def load_h5_data_multires_(h5_filename, num_points):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    sub_idx = []
    for i in range(len(num_points)):
        sub_idx.append(f['sub_idx_' + str(num_points[i])])

    class_label = f['class_label'][:]

    if 'part_label' in f:
        part_label = f['part_label'][:]
        return (data, sub_idx, part_label, class_label)
    else:
        return (data, sub_idx, class_label)

def load_h5_data_multires(data_path, files_list_path, num_points):
    files_list = [line.rstrip() for line in open(files_list_path)]
    data = []
    labels = []
    sub_idx = []
    for j in range(len(num_points)):
        sub_idx.append([])
    for i in range(len(files_list)):
        data_, sub_idx_, labels_ = load_h5_data_multires_(os.path.join(data_path, files_list[i]), num_points)
        data.append(data_)
        for j in range(len(sub_idx_)):
            sub_idx[j].append(sub_idx_[j])
        labels.append(labels_)
    data = np.concatenate(data, axis=0)
    for j in range(len(num_points)):
        sub_idx[j] = np.concatenate(sub_idx[j], axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, sub_idx, labels

class Provider(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files_list, data_path, n_points, n_samples=None,
                 batch_size=32, preprocess=list(), sampler=None, shuffle=True):
        'Initialization'

        if sampler is None:
            self.sampler = uniform_pc_sampling
        else:
            self.sampler = sampler
        self.n_samples = n_samples
        self.data = load_h5_data_files(data_path, files_list)
        self.n_points_data = np.shape(self.data)[1]
        self.n_shapes = self.data.shape[0]
        self.nv = self.data.shape[1]
        self.n_points = n_points

        self.batch_size = batch_size

        self.shuffle = shuffle
        self.preprocess = preprocess
        # self.pool = Pool(4)
        self.on_epoch_end()
        self.i_ = 0
        self.time0 = time.time()
        self.preprocess_time = 0
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_shapes / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(indexes)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_shapes)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.i_ = 0
    def __data_generation(self, indexes):
        preproc_time = time.time()
        idx = np.random.permutation(np.arange(self.n_points_data))[:self.n_points]
        Y = self.data[indexes, ...]
        Y = Y[:, idx, ...]
        for i in range(len(self.preprocess)):
            Y = pc_batch_preprocess(x=Y, f=None, y=None, proc=self.preprocess[i])
        Y = Y.astype(dtype=np.float32)
        self.preprocess_time += (time.time()-preproc_time)
        # print('time_ratio = ', self.preprocess_time / (time.time() - self.time0))
        # print(self.i_*self.batch_size)
        self.i_ += 1
        # X = self.sampler(Y, self.n_samples)
        return Y

    def set_preprocessing(self, preprocessing):
        self.preprocess = preprocessing

    def get_data(self):
        return self.data

def load_h5_data_multires_(h5_filename, num_points):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    sub_idx = []
    for i in range(len(num_points)):
        sub_idx.append(f['sub_idx_' + str(num_points[i])])

    class_label = f['class_label'][:]

    if 'part_label' in f:
        part_label = f['part_label'][:]
        return (data, sub_idx, part_label, class_label)
    else:
        return (data, sub_idx, class_label)

def load_h5_data_multires(data_path, files_list_path, num_points):
    files_list = [line.rstrip() for line in open(files_list_path)]
    data = []
    labels = []
    sub_idx = []
    for j in range(len(num_points)):
        sub_idx.append([])
    for i in range(len(files_list)):
        data_, sub_idx_, labels_ = load_h5_data_multires_(os.path.join(data_path, files_list[i]), num_points)
        data.append(data_)
        for j in range(len(sub_idx_)):
            sub_idx[j].append(sub_idx_[j])
        labels.append(labels_)
    data = np.concatenate(data, axis=0)
    for j in range(len(num_points)):
        sub_idx[j] = np.concatenate(sub_idx[j], axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, sub_idx, labels


class ClassificationProvider2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files_list, data_path, n_classes, n_points,
                 batch_size=32, preprocess=list(), shuffle=True, classes=list()):
        'Initialization'
        # self.data, self.labels = load_h5_files(data_path, files_list)
        self.data, self.sub_idx, self.labels = load_h5_data_multires(data_path, files_list, n_points)

        # for j in range()

        self.labels = np.reshape(self.labels, (-1,))
        self.n_points_data = np.shape(self.data)[1]
        self.n_shapes = self.data.shape[0]
        self.nv = self.data.shape[1]
        self.classes = classes
        self.n_points = n_points
        # self.data = index_points(self.data, self.n_points)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.preprocess = preprocess
        # self.pool = Pool(4)
        self.X = []
        for j in range(len(n_points)):
            self.X.append(np.zeros((batch_size, n_points[j], 3)))
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_shapes / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_shapes)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):

        x = self.data[indexes, ...]

        for j in range(len(self.n_points)):
            sub_idx = self.sub_idx[j][indexes, ...]
            for k in range(x.shape[0]):
                idx = sub_idx[k, ...]
                self.X[j][k, ...] = x[k, idx, ...]



        # idx = np.random.permutation(np.arange(self.n_points_data))[:self.n_points]
        # X = self.data[indexes, ...]
        # X = X[:, idx, ...]
        y = self.labels[indexes, ...]

        """
        for i in range(len(self.preprocess)):
            X, y = pc_batch_preprocess(x=X, y=y, proc=self.preprocess[i])
        """

        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        # y = np.expand_dims(y, axis=1)
        # y = np.repeat(y, axis=1, repeats=12)
        # print('y shape !! ', y.shape)
        return self.X, y

    def set_preprocessing(self, preprocessing):
        self.preprocess = preprocessing

    def get_data(self):
        return self.data, self.labels