# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:39:40 2017
@author: Gary
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Lambda
import math

import h5py
import sys
sys.path.append("../")
from auto_encoder.tfn_capsules_multi_frame import TFN_multi


def sq_dist_mat(x, y):
    r0 = tf.multiply(x, x)
    r0 = tf.reduce_sum(r0, axis=2, keepdims=True)

    r1 = tf.multiply(y, y)
    r1 = tf.reduce_sum(r1, axis=2, keepdims=True)
    r1 = tf.transpose(r1, [0, 2, 1])

    sq_distance_mat = r0 - 2. * tf.matmul(x, tf.transpose(y, [0, 2, 1])) + r1
    return sq_distance_mat


def var(x, axis_mean=0, axis_norm=1):
    mean = tf.reduce_mean(x, axis=axis_mean, keepdims=True)
    y = tf.subtract(x, mean)
    yn = tf.reduce_sum(y * y, axis=axis_norm, keepdims=True)
    yn = tf.reduce_mean(yn, axis=axis_mean, keepdims=True)
    return yn, mean

def std(x, axis_mean=0, axis_norm=1):
    yn, mean = var(x, axis_mean=axis_mean, axis_norm=axis_norm)
    return tf.sqrt(yn), mean

def pca_align(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    centred_x = tf.subtract(x, c)
    covar_mat = tf.reduce_mean(tf.einsum('bvi,bvj->bvij', centred_x, centred_x), axis=1, keepdims=False)
    _, v = tf.linalg.eigh(covar_mat)

    x = tf.einsum('bij,bvi->bvj', v, centred_x)
    return x

def tf_center(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    return tf.subtract(x, c)

def tf_dist(x, y):
    d = tf.subtract(x, y)
    d = tf.multiply(d, d)
    d = tf.reduce_sum(d, axis=-1, keepdims=False)
    d = tf.sqrt(d + 0.00001)
    return tf.reduce_mean(d)

"""
def rotate_point_cloud(arr):
    return np.einsum('ij,vj->vi', generate_3d(), arr)
"""


def tf_random_rotation(shape):
    if isinstance(shape, int):
        shape = [shape]

    batch_size_ = shape[0]
    t = tf.random.uniform(shape + [3], minval=0., maxval=1.)
    c1 = tf.cos(2 * np.pi * t[:, 0])
    s1 = tf.sin(2 * np.pi * t[:, 0])

    c2 = tf.cos(2 * np.pi * t[:, 1])
    s2 = tf.sin(2 * np.pi * t[:, 1])

    z = tf.zeros(shape)
    o = tf.ones(shape)

    R = tf.stack([c1, s1, z, -s1, c1, z, z, z, o], axis=-1)
    R = tf.reshape(R, shape + [3, 3])

    v1 = tf.sqrt(t[:, -1])
    v3 = tf.sqrt(1-t[:, -1])
    v = tf.stack([c2 * v1, s2 * v1, v3], axis=-1)
    H = tf.tile(tf.expand_dims(tf.eye(3), axis=0), (batch_size_, 1, 1)) - 2.* tf.einsum('bi,bj->bij', v, v)
    M = -tf.einsum('bij,bjk->bik', H, R)
    return M

def tf_random_rotate(x):
    R = tf_random_rotation(x.shape[0])
    return tf.einsum('bij,bpj->bpi', R, x)
def diameter(x, axis=-2, keepdims=True):
    return tf.reduce_max(x, axis=axis, keepdims=keepdims) - tf.reduce_min(x, axis=axis, keepdims=keepdims)

def Log2(x):
    return (math.log10(x) / math.log10(2))
def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) == math.floor(Log2(n)))

def kdtree_indexing(x, depth=None):
    num_points = x.shape[1]
    assert isPowerOfTwo(num_points)
    if depth is None:
        depth = int(np.log(num_points) / np.log(2.) + 0.1)
    y = x
    batch_idx = tf.range(x.shape[0],dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, (-1, 1))
    batch_idx = tf.tile(batch_idx, (1, x.shape[1]))

    for i in range(depth):
        y_shape = list(y.shape)
        diam = diameter(y)
        split_idx = tf.argmax(diam, axis=-1, output_type=tf.int32)
        split_idx = tf.tile(split_idx, (1, y.shape[1]))
        # split_idx = tf.tile(split_idx, (1, y.shape[1], 1))
        idx = tf.range(y.shape[0])
        idx = tf.expand_dims(idx, axis=-1)
        idx = tf.tile(idx, (1, y.shape[1]))
        branch_idx = tf.range(y.shape[1])
        branch_idx = tf.expand_dims(branch_idx, axis=0)
        branch_idx = tf.tile(branch_idx, (y.shape[0], 1))
        split_idx = tf.stack([idx, branch_idx, split_idx], axis=-1)
        m = tf.gather_nd(y, split_idx)
        sort_idx = tf.argsort(m, axis=-1)
        sort_idx = tf.stack([idx, sort_idx], axis=-1)
        y = tf.gather_nd(y, sort_idx)
        y = tf.reshape(y, (-1, int(y.shape[1] // 2), 3))

    y = tf.reshape(y, x.shape)
    return y

def normalize(x):
    s, m = std(x, axis_mean=1, axis_norm=-1)
    x = tf.divide(tf.subtract(x, m), s)
    return x

def orth_procrustes(x, y):
    x = normalize(tf_center(x))
    y = normalize(tf_center(y))
    xty = tf.einsum('bvi,bvj->bij', y, x)
    s, u, v = tf.linalg.svd(xty)
    r = tf.einsum('bij,bkj->bik', u, v)
    return r

def mat_mul(A, B):
    return tf.matmul(A, B)


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def shuffle(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return np.take(x, idx, axis=0), np.take(y, idx, axis=0)

def shuffle_list(x):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    y = []
    for i in range(len(x)):
        y.append(np.take(x[i], idx, axis=0))
    return y

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

def tf_jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    jittered_data = tf.clip_by_value(sigma * tf.random.normal(batch_data.shape), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

def compute_keypoints(A, x, eps=1e-6):
    A_sum = tf.expand_dims(tf.maximum(tf.reduce_sum(A, axis=1, keepdims=True), eps), axis=-1)
    x_sum = tf.multiply(tf.expand_dims(A, axis=-1), tf.expand_dims(x, axis=-2))
    x_sum = tf.reduce_sum(x_sum, axis=1, keepdims=True)
    theta = tf.divide(x_sum, A_sum)
    return theta

def key_points_embedding(A, x, eps=1e-6):
    theta = compute_keypoints(A, x, eps=eps)
    y_ = tf.expand_dims(x, axis=-2)
    y = tf.subtract(y_, theta)
    y = tf.concat([y_, y], axis=-2)
    # y = tf.reshape(y, (y.shape[0], y.shape[1], -1))
    return y

def loadh5(path):
    fx_input = h5py.File(path, 'r')
    x = fx_input['data'][:]
    fx_input.close()
    return x

num_points = 1024
k = 40
batch_size = 32
def build_model(train, canonicalize=True):
    # number of points in each sample


    # number of categories


    # define optimizer
    # adam = optimizers.Adam(lr=0.001, decay=0.7)

    # ------------------------------------ TFN canonicalizer

    input_points = Input(shape=(num_points, 3), batch_size=batch_size)

    x = input_points
    if train:
        x = tf_jitter_point_cloud(x)
        # can_points = tf.expand_dims(can_points, axis=-2)

    x_ = x
    # can_points_1 = tf.reshape(can_points, (can_points.shape[0], can_points.shape[1], -1))
    x = Convolution1D(64, 1, activation='relu',
                  input_shape=(num_points, 3))(x)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=num_points)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    # forward net
    # g = Lambda(mat_mul, arguments={'B': input_T})(can_points)
    g = tf.einsum("bij,bvj->bvi", input_T, x_)
    # g = tf.reshape(g, (g.shape[0], g.shape[1], -1))
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)

    # feature transform net
    f = Convolution1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(1024, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=num_points)(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    # g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    g = tf.einsum("bij,bvj->bvi", feature_T, g)
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global_feature
    global_feature = MaxPooling1D(pool_size=num_points)(g)

    # point_net_cls
    c = Dense(512, activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.7)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.7)(c)
    c = Dense(k, activation='softmax')(c)
    prediction = Flatten()(c)
    # --------------------------------------------------end of pointnet

    # print the model summary
    model = Model(inputs=input_points, outputs=prediction)
    return model







train_model = build_model(True)
test_model = build_model(False)
test_model.set_weights(train_model.get_weights())
print(train_model.summary())

# load input rotations
input_rotations_path = "/gpfs/scratch/rsajnani/rsajnani/research/evaluation/rotations.h5"
input_rotations = loadh5(input_rotations_path)
# load train points and labels
path = "/gpfs/scratch/rsajnani/rsajnani/research/dataset/modelnet40_hdf5_1024_original/data_hdf5/"
can_frames_path = "/gpfs/scratch/rsajnani/rsajnani/research/evaluation/classification/CaCa_modelnet40/"
train_path = path
filenames = ["train_data_0.h5", "train_data_1.h5", "train_data_2.h5", "train_data_3.h5", "train_data_4.h5"]
print(train_path)
print(filenames)
train_points = None
train_labels = None
train_can_frames = None
for d in filenames:
    cur_points, cur_labels = load_h5(os.path.join(train_path, d))
    cur_can_frames = loadh5(os.path.join(can_frames_path, d))
    cur_can_frames = cur_can_frames.reshape(1, -1, 3, 3)
    cur_points = cur_points.reshape(1, -1, 3)
    cur_labels = cur_labels.reshape(1, -1)
    if train_labels is None or train_points is None:
        train_labels = cur_labels
        train_points = cur_points
        train_can_frames = cur_can_frames
    else:
        train_labels = np.hstack((train_labels, cur_labels))
        train_points = np.hstack((train_points, cur_points))
        train_can_frames = np.hstack((train_can_frames, cur_can_frames))

train_can_frames = train_can_frames.reshape(-1, 1, 3, 3)
train_points_r = train_points.reshape(-1, num_points, 3)
train_labels_r = train_labels.reshape(-1, 1)
train_can_frames = train_can_frames.reshape(-1, 1, 3, 3)

train_points_r = np.einsum("bij,bvj->bvi", train_can_frames[:, 0, ...], train_points_r)

# load test points and labels"
test_path = path
filenames = ["test_data_0.h5", "test_data_1.h5"]
print(test_path)
print(filenames)
test_points = None
test_can_frames = None
test_labels = None
for d in filenames:
    cur_points, cur_labels = load_h5(os.path.join(test_path, d))
    cur_can_frames = loadh5(os.path.join(can_frames_path, d))
    cur_can_frames = cur_can_frames.reshape(1, -1, 3, 3)
    cur_points = cur_points.reshape(1, -1, 3)
    cur_labels = cur_labels.reshape(1, -1)
    if test_labels is None or test_points is None:
        test_labels = cur_labels
        test_points = cur_points
        test_can_frames = cur_can_frames
    else:
        test_labels = np.hstack((test_labels, cur_labels))
        test_points = np.hstack((test_points, cur_points))
        test_can_frames = np.hstack((test_can_frames, cur_can_frames))

test_can_frames = test_can_frames.reshape(-1, 128, 3, 3)
test_points_r = test_points.reshape(-1, num_points, 3)
test_labels_r = test_labels.reshape(-1, 1)
test_can_frames = test_can_frames.reshape(-1, 128, 3, 3)

test_points_r_ = np.einsum("bij,bvj->bvi", test_can_frames[:, 0, ...], test_points_r)
# label to categorical

Y_train = tf.keras.utils.to_categorical(train_labels_r, k)
Y_test = tf.keras.utils.to_categorical(test_labels_r, k)

# compile classification model
train_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

test_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])




def aligned_kdtree_indexing(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    centred_x = tf.subtract(x, c)
    covar_mat = tf.reduce_mean(tf.einsum('bvi,bvj->bvij', centred_x, centred_x), axis=1, keepdims=False)
    _, v = tf.linalg.eigh(covar_mat)

    x = tf.einsum('bij,bvi->bvj', v, centred_x)
    x = tf.add(x, c)
    return kdtree_indexing(x)

def aligned_kdtree_indexing_(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    centred_x = tf.subtract(x, c)
    covar_mat = tf.reduce_mean(tf.einsum('bvi,bvj->bvij', centred_x, centred_x), axis=1, keepdims=False)
    _, v = tf.linalg.eigh(covar_mat)

    x = tf.einsum('bij,bvi->bvj', v, centred_x)
    x = tf.add(x, c)
    y, points_idx, points_idx_inv = kdtree_indexing_(x)
    return y, points_idx, points_idx_inv, v



J = train_points_r.shape[0] - (train_points_r.shape[0] % batch_size)
J_test = test_points_r.shape[0] - (test_points_r.shape[0] % batch_size)

# Fit model on training data
for i in range(1,200):
    #model.fit(train_points_r, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
    # rotate and jitter the points
    # train_points_rotate = rotate_point_cloud(train_points_r)
    # train_points_jitter = jitter_point_cloud(train_points_rotate)
    train_points_shuffle, Y_train_shuffle = shuffle(train_points_r, Y_train)
    train_points_shuffle = train_points_shuffle[:J, ...]
    Y_train_shuffle = Y_train_shuffle[:J, ...]
    train_model.fit(train_points_shuffle, Y_train_shuffle, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
    s = 'Current epoch is:' + str(i)
    print(s)
    if i % 5 == 0:
        test_model.set_weights(train_model.get_weights())

        score = test_model.evaluate(test_points_r_[:J_test, ...], Y_test[:J_test, ...], verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

# save model:
weights_path = "./model_classification/CaCa/"
test_model.save_weights(os.path.join(weights_path, "CaCa_pointnet.h5"))

test_model.load_weights("./model_classification/CaCa/CaCa_pointnet.h5")
#print()
# score the model
s = 0.
l = 0.

for i in range(test_can_frames.shape[1]):
    frames = np.einsum("bij,jk->bik", test_can_frames[:, i, ...], input_rotations[i, ...])
    test_points__ = np.einsum("bij,bvj->bvi", frames, test_points_r)
    test_points__, Y_test_ = shuffle(test_points__, Y_test)
    test_points__ = test_points__[:J_test, ...]
    Y_test_ = Y_test_[:J_test, ...]
    score = test_model.evaluate(test_points__, Y_test_, verbose=1)
    l += score[0]
    s += score[1]

l /= test_can_frames.shape[1]
s /= test_can_frames.shape[1]
print('Test loss: ', l)
print('Test accuracy: ', s)
