import tensorflow as tf
from spherical_harmonics.kernels import tf_eval_monom_basis, tf_monom_basis_offset, tf_monomial_basis_3D_idx, compute_monomial_basis_offset
from spherical_harmonics.kernels import real_spherical_harmonic, zernike_kernel_3D, tf_zernike_kernel_basis
from spherical_harmonics.kernels import A, B, associated_legendre_polynomial
from utils.pointclouds_utils import generate_3d
from network_utils.group_points import GroupPoints
from network_utils.sparse_grid_sampling_eager import GridSampler, GridPooling, extract_batch_idx
from network_utils.pooling import kd_pooling_1d, kd_median_sampling, kdtree_indexing, aligned_kdtree_indexing
from utils.pointclouds_utils import np_kd_tree_idx, pc_batch_preprocess
from data_providers.provider import Provider
from data_providers.classification_datasets import datsets_list
from spherical_harmonics.clebsch_gordan_decomposition import tf_clebsch_gordan_decomposition
from network_utils.pooling import extract_samples_slices
from time import time
from sklearn.neighbors import NearestNeighbors
# from unocs.train import lexicographic_ordering

from auto_encoder.tfn_auto_encoder_svd import TFN
from utils.losses import chamfer_distance_l2
from plyfile import PlyData, PlyElement





import h5py
"""
from sympy import *
x, y, z = symbols("x y z")
"""


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

print(tf.__version__)
import numpy as np
from utils.data_prep_utils import load_h5_data
from pooling import GridBatchSampler
from utils.pointclouds_utils import setup_pcl_viewer
import vispy
from activations import tf_dodecahedron_sph
from spherical_harmonics.kernels import monomial_basis_3D
from pooling import simple_grid_sampler

def tf_center(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    return tf.subtract(x, c)

def tf_random_rotation(shape):
    if isinstance(shape, int):
        shape = [shape]

    batch_size = shape[0]
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
    H = tf.tile(tf.expand_dims(tf.eye(3), axis=0), (batch_size, 1, 1)) - 2.* tf.einsum('bi,bj->bij', v, v)
    M = -tf.einsum('bij,bjk->bik', H, R)
    return M

def tf_random_rotate(x):
    R = tf_random_rotation(x.shape[0])
    return tf.einsum('bij,bpj->bpi', R, x)

def sigma(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    x = tf.subtract(x, c)
    n2 = tf.multiply(x, x)
    n2 = tf.reduce_sum(n2, axis=-1, keepdims=True)
    n2 = tf.reduce_mean(n2, axis=1, keepdims=True)
    s = tf.sqrt(n2 + 0.000000001)
    return s

def var_normalize(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    x = tf.subtract(x, c)
    n2 = tf.multiply(x, x)
    n2 = tf.reduce_sum(n2, axis=-1, keepdims=True)
    n2 = tf.reduce_mean(n2, axis=1, keepdims=True)
    s = tf.sqrt(n2 + 0.000000001)
    x = tf.divide(x, s)
    return x




def registration(x, y):
    x = var_normalize(x)
    y = var_normalize(y)
    xyt = tf.einsum('bvi,bvj->bij', x, y)
    s, u, v = tf.linalg.svd(xyt)
    R = tf.matmul(v, u, transpose_b=True)


    d = tf.linalg.det(R)
    o = tf.ones([R.shape[0], 2])
    d_ = tf.concat([o, tf.expand_dims(d, axis=-1)], axis=-1)
    d_ = tf.expand_dims(d_, axis=1)
    v = tf.multiply(v, d_)
    R = tf.matmul(v, u, transpose_b=True)
    # return R, d

    return R

def lexicographic_ordering(x):
    m = tf.reduce_min(x, axis=1, keepdims=True)
    y = tf.subtract(x, m)
    M = tf.reduce_max(y, axis=1, keepdims=True)
    y = tf.multiply(M[..., 2]*M[..., 1], y[..., 0]) + tf.multiply(M[..., 2], y[..., 1]) + y[..., 2]
    batch_idx = tf.range(y.shape[0])
    batch_idx = tf.expand_dims(batch_idx, axis=-1)
    batch_idx = tf.tile(batch_idx, multiples=(1, y.shape[1]))
    idx = tf.argsort(y, axis=1)
    idx = tf.stack([batch_idx, idx], axis=-1)
    return tf.gather_nd(x, idx)



def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


NUM_POINTS = 1024

BATCH_SIZE = 32
SHUFFLE = True

def load_dataset(dataset):
    batch_size = BATCH_SIZE
    num_points = NUM_POINTS

    train_files_list = dataset['train_files_list']
    val_files_list = dataset['val_files_list']
    test_files_list = dataset['test_files_list']

    train_data_folder = dataset['train_data_folder']
    val_data_folder = dataset['val_data_folder']
    test_data_folder = dataset['test_data_folder']

    train_preprocessing = dataset['train_preprocessing']
    val_preprocessing = dataset['val_preprocessing']
    test_preprocessing = dataset['test_preprocessing']
    NUM_SAMPLES_DOWNSAMPLED = None
    """
    train_provider = Provider(files_list=train_files_list,
                              data_path=train_data_folder,
                              n_points=num_points,
                              n_samples=NUM_SAMPLES_DOWNSAMPLED,
                              batch_size=batch_size,
                              preprocess=train_preprocessing,
                              shuffle=SHUFFLE)

    val_provider = Provider(files_list=val_files_list,
                            data_path=val_data_folder,
                            n_points=num_points,
                            n_samples=NUM_SAMPLES_DOWNSAMPLED,
                            batch_size=batch_size,
                            preprocess=val_preprocessing,
                            shuffle=SHUFFLE)
    """

    test_provider = Provider(files_list=test_files_list,
                             data_path=test_data_folder,
                             n_points=num_points,
                             n_samples=NUM_SAMPLES_DOWNSAMPLED,
                             batch_size=batch_size,
                             preprocess=test_preprocessing,
                             shuffle=True)

    return test_provider




def save_obj( x, filename: str ):
    """Saves a WavefrontOBJ object to a file

    Warning: Contains no error checking!

    """
    with open( filename, 'w' ) as ofile:
        for i in range(x.shape[0]):
            xi = [x[i, 0], x[i, 1], x[i, 2]]
            ofile.write('v ' + ' '.join(['{}'.format(v) for v in xi]) + '\n')




test_provider = load_dataset(datsets_list[0])



inputs = tf.keras.layers.Input(batch_shape=(BATCH_SIZE, NUM_POINTS, 3))


autoencoder = tf.keras.models.Model(inputs=inputs, outputs=TFN(1024)(inputs), trainable=False)


print(autoencoder.layers)

# weights = 'ckpt-2.data-00000-of-00001'
autoencoder.load_weights('E:/Users/Adrien/Documents/results/pose_canonicalization_tfn/weights_0.h5')


"""
h5_filename = "E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_classes/data_hdf5/test_toilet.h5"

# h5_filename = "E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_original/data_hdf5/test_data_0.h5"

f = h5py.File(h5_filename, mode='r')
data = f['data'][:]

# data, labels = load_h5(h5_filename)

print(data.shape)
# print(labels.shape)
"""

# x = data[17, ...]

def select_sym(x, y):
    sx = sigma(x)
    X = var_normalize(x)
    Y = var_normalize(y)
    X0 = tf.stack([-X[..., 0], X[..., 1], X[..., 2]], axis=-1)
    X1 = tf.stack([X[..., 0], -X[..., 1], X[..., 2]], axis=-1)
    X2 = tf.stack([X[..., 0], X[..., 1], -X[..., 2]], axis=-1)
    R = registration(X, Y)
    R0 = registration(X0, Y)
    R1 = registration(X1, Y)
    R2 = registration(X2, Y)
    R = tf.stack([R, R0, R1, R2], axis=1)
    X = tf.stack([X, X0, X1, X2], axis=1)
    RX = tf.einsum("bkij,bkvj->bkvi", R, X)
    n2 = tf.subtract(RX, tf.expand_dims(Y, axis=1))
    n2 = tf.multiply(n2, n2)
    n2 = tf.reduce_sum(n2, axis=-1, keepdims=False)
    n2 = tf.reduce_mean(n2, axis=2, keepdims=False)
    idx = tf.argmin(n2, axis=1)
    batch_idx = tf.range(x.shape[0], dtype=tf.int64)
    print(batch_idx.shape)
    print(idx.shape)
    idx = tf.stack([batch_idx, idx], axis=-1)
    x = tf.gather_nd(X, idx)
    x = tf.multiply(sx, x)
    return x

R = tf_random_rotation(1)
R = R[0]

@tf.function
def test_step(x):
    R = tf_random_rotation(x.shape[0])
    # x = tf_random_rotate(x)
    x = tf.einsum('bij,bpj->bpi', R, x)
    x = kdtree_indexing(x)
    # y = tf.einsum('ij,bpj->bpi', R, x)
    _, y, _ = autoencoder(x)

    """
    x = var_normalize(x)
    y = var_normalize(y)

    R_ = registration(x, y)
    y = tf.einsum('bij,bpj->bpi', R_, x)
    x = tf.einsum('bji,bpj->bpi', R, x)
    return x, y, d
    """
    return y

EPOCHS = 4
Y = []
X = []
for epoch in range(EPOCHS):
    test_provider.on_epoch_end()
    for x in test_provider:
        print("u")
        y = test_step(x)
        # print(d)
        Y.append(y)
        X.append(x)

L = len(X)
X = tf.concat(X, axis=0)
Y = tf.concat(Y, axis=0)
# X = var_normalize(X)
# Y = var_normalize(Y)

# X = select_sym(X, Y)

R = registration(tf.reshape(X, [1, -1, 3]), tf.reshape(Y, [1, -1, 3]))
# R = registration(tf.expand_dims(X[0, ...], axis=0), tf.expand_dims(Y[0, ...], axis=0))
R = R[0, ...]


# X0 = tf.stack([-X[..., 0], X[..., 1], X[..., 2]], axis=-1)
# X1 = tf.stack([X[..., 0], -X[..., 1], X[..., 2]], axis=-1)
# X2 = tf.stack([X[..., 0], X[..., 1], -X[..., 2]], axis=-1)
# X = tf.stack([X, X0, X1, X2], axis=1)

sX = sigma(X)
X = var_normalize(X)
X = tf.multiply(sX, X)
Y = var_normalize(Y)
Y = tf.multiply(sX, Y)

RX = tf.einsum("ij,bvj->bvi", R, X)
# n2 = tf.subtract(RX, tf.expand_dims(Y, axis=1))
n2 = tf.subtract(RX, Y)
n2 = tf.multiply(n2, n2)
n2 = tf.reduce_sum(n2, axis=-1, keepdims=False)
mean_root_square = tf.reduce_sum(n2, axis=1, keepdims=False)
mean_root_square = tf.sqrt(mean_root_square + 0.000000001) / X.shape[1]
# mean_root_square = tf.reduce_min(mean_root_square, axis=1, keepdims=False)
mean_root_square = tf.reduce_mean(mean_root_square)
print("mrs_loss ", float(mean_root_square))


k = 0.
l = 0.

for i in range(L):
    x = RX[i*BATCH_SIZE:(i+1)*BATCH_SIZE, ...]
    y = Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE, ...]
    l += float(chamfer_distance_l2(x, y))
    k += 1.

print("chamfer loss ", l / k)
