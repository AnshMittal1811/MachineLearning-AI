import numpy as np
import tensorflow as tf
from utils.data_prep_utils import load_h5_files
from utils.pointclouds_utils import pc_batch_preprocess
import h5py
import os
import json

class SegmentationProvider(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files_list, data_path, n_parts, n_classes, n_points, samples_ratio=1.,
                 batch_size=32, preprocess=list(), shuffle=True, parts=list(), labels_to_cat=None):
        'Initialization'
        # self.data, self.labels = load_h5_files(data_path, files_list)

        self.labels_to_cat = None
        self.cat_to_labels = None
        self.seg_parts = None

        if labels_to_cat is not None:
            with open(labels_to_cat) as f:
                cat_labels = json.load(f)
            self.labels_to_cat = ['']*len(cat_labels)
            self.cat_to_labels = dict()
            for i in range(len(cat_labels)):
                self.labels_to_cat.append(cat_labels[i][0])
                if cat_labels[i][0] in self.cat_to_labels:
                    self.cat_to_labels[cat_labels[i][0]].append(i)
                else:
                    self.cat_to_labels[cat_labels[i][0]] = [i]

            self.seg_parts = self.cat_to_labels
        else:
            self.cat_to_labels = {'0': range(n_parts)}
            self.seg_parts = self.cat_to_labels
            """
            for cat in self.cat_to_labels.keys():
                if cat in self.seg_parts:
                    self.seg_parts[cat].append(self.cat_to_labels[cat])
                else:
                    self.seg_parts[cat] = [self.cat_to_labels[cat]]
            """


        data = []
        part_labels = []
        class_labels = []
        files_list = [line.rstrip() for line in open(files_list)]
        for file in files_list:
            f = h5py.File(os.path.join(data_path, file))
            data.append(f['data'][:])
            part_labels.append(f['pid'][:].astype(np.int32))
            class_labels.append(f['label'][:].astype(np.int32))

        self.data = np.concatenate(data, axis=0)
        self.part_labels = np.concatenate(part_labels, axis=0)
        self.part_labels = np.expand_dims(self.part_labels, axis=-1)
        self.class_labels = np.concatenate(class_labels, axis=0)

        self.part_weights = np.zeros((n_classes, n_parts, ))
        self.nb_seen_per_class = np.zeros((n_classes, 1))



        for i in range(self.data.shape[0]):
            one_hot_labels = self.part_labels[i, ...]
            one_hot_labels = np.expand_dims(one_hot_labels, axis=0)
            one_hot_labels = tf.keras.utils.to_categorical(one_hot_labels, n_parts)
            self.nb_seen_per_class[self.class_labels[i], 0] += 1.
            self.part_weights[self.class_labels[i], :] += np.sum(one_hot_labels[0, :, :], axis=0)

        self.part_weights = np.divide(self.part_weights, self.nb_seen_per_class)
        self.part_weights = np.sum(self.part_weights, axis=0, keepdims=False)/float(self.data.shape[1])

        self.part_weights = np.tile(np.expand_dims(self.part_weights, axis=0), reps=(n_points, 1))

        print('weights ', self.part_weights)

        self.n_points_data = np.shape(self.data)[1]
        self.n_samples = self.data.shape[0]
        self.nv = self.data.shape[1]
        self.parts = parts
        self.n_points = n_points
        self.samples_ratio = samples_ratio
        self.n_volume_samples = int(self.samples_ratio * self.n_points)
        self.batch_size = batch_size
        self.n_parts = n_parts
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):

        idx = np.random.permutation(np.arange(self.n_points_data))[:self.n_points]
        # samples_idx = np.random.permutation(np.arange(self.n_points))[:self.n_points]

        class_labels = self.class_labels[indexes, ...]
        class_labels = tf.keras.utils.to_categorical(class_labels, num_classes=self.n_classes)
        X = self.data[indexes, ...]
        X = X[:, idx, ...]
        y = self.part_labels[indexes, ...]
        y = y[:, idx, ...]
        for i in range(len(self.preprocess)):
            X, y = pc_batch_preprocess(X, y=y, proc=self.preprocess[i])

        # x = X[:, samples_idx, ...]
        # y = y[: samples_idx, ...]
        # y = tf.keras.utils.to_categorical(y, num_classes=self.n_parts)
        return [X, class_labels], y

    def set_preprocessing(self, preprocessing):
        self.preprocess = preprocessing

    def get_data(self):
        return self.data, self.part_labels, self.class_labels

    def get_batch_size(self):
        return self.batch_size

    def get_num_parts(self):
        return self.n_parts

    def get_num_classes(self):
        return self.n_classes

    def get_num_points(self):
        return self.n_points

    def get_preprocess(self):
        return self.preprocess