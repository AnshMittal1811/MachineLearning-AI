import tensorflow as tf
import math
from tensorflow.python.keras.layers import AveragePooling2D, AveragePooling1D, MaxPooling1D, MaxPooling2D
import numpy as np
import h5py

"""
computes unique elements y in a 1D tensor x
and idx such that x[i] = y[idx[i]]
and inverse indices such that idx[idx_inv[j]] = j
"""
def tf_unique_with_inverse(x):
    y, idx = tf.unique(x)
    num_segments = y.shape[0]
    num_elems = x.shape[0]
    return (y, idx,  tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments))

def grid_sampler_(x, cell_size, num_points_target):
    batch_size = x.shape[0]
    num_points = x.shape[1]
    y = tf.round(x / cell_size)
    grid_idx = tf.cast(y, dtype=tf.int32)
    y = cell_size * y
    min_grid_idx = tf.reduce_min(grid_idx)
    grid_size = tf.reduce_max(grid_idx) - min_grid_idx + 2
    grid_idx = grid_idx - min_grid_idx
    linear_idx = grid_size * (grid_size * grid_idx[..., 0] + grid_idx[..., 1]) + grid_idx[..., 2]
    batch_idx = tf.expand_dims(tf.multiply(grid_size ** 3, tf.range(batch_size)), axis=-1)
    linear_idx = tf.add(batch_idx, linear_idx)
    linear_idx = tf.reshape(linear_idx, [-1])
    ul, ul_idx, ul_idx_inv = tf_unique_with_inverse(linear_idx)

    # pad_size = linear_idx.shape[0] - ul_idx_inv.shape[0]
    # idx = tf.pad(x, paddings=[[0, pad_size]], mode="CONSTANT", constant_values=-1)
    mask = tf.scatter_nd(indices=tf.expand_dims(ul_idx_inv, axis=-1),
                         updates=tf.ones(ul_idx_inv.shape), shape=linear_idx.shape)
    mask = tf.reshape(mask, (batch_size, num_points))
    idx = tf.argsort(mask, axis=-1, direction='DESCENDING')
    idx = idx[:, :num_points_target]
    batch_idx = tf.range(batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    batch_idx = tf.tile(batch_idx, (1, num_points_target))
    idx = tf.stack([batch_idx, idx], axis=-1)
    y = tf.gather_nd(y, idx)
    # y = y[:, :self.num_points_target, :]
    mask = tf.gather_nd(mask, idx)
    # mask = mask[:, :self.num_points_target, :]
    return y, idx, mask, ul_idx, ul_idx_inv

def grid_sampler(x, cell_size, num_points_target):
    batch_size = x.shape[0]
    num_points = x.shape[1]
    y = tf.round(x / cell_size)
    grid_idx = tf.cast(y, dtype=tf.int32)
    y = cell_size * y
    min_grid_idx = tf.reduce_min(grid_idx)
    grid_size = tf.reduce_max(grid_idx) - min_grid_idx + 2
    grid_idx = grid_idx - min_grid_idx
    linear_idx = grid_size * (grid_size * grid_idx[..., 0] + grid_idx[..., 1]) + grid_idx[..., 2]
    batch_idx = tf.expand_dims(tf.multiply(grid_size ** 3, tf.range(batch_size)), axis=-1)
    linear_idx = tf.add(batch_idx, linear_idx)
    linear_idx = tf.reshape(linear_idx, [-1])
    ul, ul_idx, ul_idx_inv = tf_unique_with_inverse(linear_idx)

    # pad_size = linear_idx.shape[0] - ul_idx_inv.shape[0]
    # idx = tf.pad(x, paddings=[[0, pad_size]], mode="CONSTANT", constant_values=-1)
    mask = tf.scatter_nd(indices=tf.expand_dims(ul_idx_inv, axis=-1),
                         updates=tf.ones(ul_idx_inv.shape), shape=linear_idx.shape)
    mask = tf.reshape(mask, (batch_size, num_points))
    idx = tf.argsort(mask, axis=-1, direction='DESCENDING')
    idx = idx[:, :num_points_target]
    batch_idx = tf.range(batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    batch_idx = tf.tile(batch_idx, (1, num_points_target))
    idx = tf.stack([batch_idx, idx], axis=-1)
    y = tf.gather_nd(y, idx)
    # y = y[:, :self.num_points_target, :]
    mask = tf.gather_nd(mask, idx)
    # mask = mask[:, :self.num_points_target, :]
    return y, idx, mask, ul_idx, ul_idx_inv

class GridBatchSampler(tf.keras.layers.Layer):
    def __init__(self, cell_size, num_points_target):
        super(GridBatchSampler, self).__init__()
        self.cell_size = cell_size
        self.num_points_target = num_points_target

    def build(self, input_shape):
        super(GridBatchSampler, self).build(input_shape)

    def call(self, x):
        return grid_sampler(x, self.cell_size, self.num_points_target)

"""

"""
def grid_pooling(x, idx, mask, ul_idx, ul_idx_inv, mode='avg'):
    batch_size = x.shape[0]
    num_points = x.shape[1]
    x_shape = x.shape
    x = tf.reshape(x, shape=(batch_size*num_points, -1))
    num_segments = tf.shape(ul_idx_inv)
    if mode == 'max':
        y = tf.math.unsorted_segment_max(x, ul_idx, num_segments)
    else:
        y = tf.math.unsorted_segment_mean(x, ul_idx, num_segments)
    y = tf.scatter_nd(indices=tf.expand_dims(ul_idx_inv, axis=-1),
                      updates=y, shape=x.shape)
    y = tf.reshape(y, x_shape)
    y = tf.gather_nd(y, idx)
    m_shape = [1]*len(x_shape)
    for i in range(len(mask.shape)):
        m_shape[i] = mask.shape[i]
    mask = tf.reshape(mask, x)
    y = tf.multiply(mask, y)
    return y

def simple_grid_sampler(x, cell_size, pool_size):
    batch_size = x.shape[0]
    num_points = x.shape[1]
    y = tf.round(x / cell_size)
    grid_idx = tf.cast(y, dtype=tf.int32)
    y = cell_size * y
    min_grid_idx = tf.reduce_min(grid_idx, axis=[1, 2], keepdims=True)
    grid_size = tf.reduce_max(grid_idx, axis=[1, 2], keepdims=True) - min_grid_idx + 2
    grid_size = grid_size[..., 0]


    grid_idx = grid_idx - min_grid_idx
    linear_idx = grid_size * (grid_size * grid_idx[..., 0] + grid_idx[..., 1]) + grid_idx[..., 2]
    idx = tf.argsort(linear_idx, axis=1)
    sample_idx = tf.range(start=0, limit=num_points, delta=pool_size)
    idx = tf.gather(idx, sample_idx, axis=1)
    batch_idx = tf.range(batch_size)
    batch_idx = tf.reshape(batch_idx, (-1, 1))
    batch_idx = tf.tile(batch_idx, (1, idx.shape[1]))
    idx = tf.stack([batch_idx, idx], axis=-1)
    return tf.gather_nd(y, idx)

def get_real_median(v):
    # v = tf.reshape(v, [-1])
    l = v.shape[-1]
    if l == 1:
        return v
    else:
        mid = l//2 + 1
        val = tf.nn.top_k(v, mid).values
        if l % 2 == 1:
            return val[-1]
        else:
            return 0.5 * (val[..., -1] + val[..., -2])

def kd_median_sampling(x, pool_size):
    y = tf.reshape(x, (x.shape[0], -1, pool_size, 3))
    m = []
    for i in range(3):
        m.append(get_real_median(y[..., i]))
    return tf.stack(m, axis=-1)

class GridBatchPooling(tf.keras.layers.Layer):
    def __init__(self, mode='avg'):
        super(GridBatchPooling, self).__init__()
        self.mode = mode

    def build(self, input_shape):
        super(GridBatchPooling, self).build(input_shape)

    def call(self, x):
        return grid_pooling(x[0], x[1], x[2], x[3], x[4], mode='avg')


def Log2(x):
    return (math.log10(x) / math.log10(2))
def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) == math.floor(Log2(n)))

"""
def kd_pooling_1d(x, pool_size, pool_mode='avg'):
    assert (isPowerOfTwo(pool_size))
    pool_size = pool_size
    if pool_mode is 'max':
        pool = MaxPooling1D(pool_size)
    else:
        pool = AveragePooling1D(pool_size)

    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = pool(x[i])
    else:
        x = pool(x)
    return x
"""

def kd_pooling_1d(x, pool_size, pool_mode='avg'):

    #assert (isPowerOfTwo(pool_size))
    pool_size = pool_size
    if pool_mode == 'max':
        pool = MaxPooling1D(pool_size)
    else:
        pool = AveragePooling1D(pool_size)
    if isinstance(x, list):
        y = []
        for i in range(len(x)):
            x.append(pool(x[i]))
    elif isinstance(x, dict):
        y = dict()
        for l in x:
            if isinstance(l, int):
                y[l] = pool(x[l])
    else:
        y = pool(x)
    return y

def kd_pooling_2d(x, pool_size, pool_mode='avg'):

    assert (isPowerOfTwo(pool_size))
    pool_size = pool_size
    if pool_mode == 'max':
        pool = MaxPooling2D((pool_size, 1))
    else:
        pool = AveragePooling2D((pool_size, 1))
    if isinstance(x, list):
        y = []
        for i in range(len(x)):
            x.append(pool(x[i]))
    elif isinstance(x, dict):
        y = dict()
        for l in x:
            if isinstance(l, int):
                y[l] = pool(x[l])
    else:
        y = pool(x)
    return y

def equivariant_kd_pooling(y, pool_size, alpha=1):
    assert(isinstance(y, dict))
    z = dict()
    for l in y:
        if l.isnumeric():
            ynl = tf.reduce_sum(tf.multiply(y[l], y[l]), axis=-2, keepdims=True)
            ynl = tf.exp(alpha*ynl)
            yl = tf.multiply(ynl, y[l])
            if isinstance(pool_size, int):
                ynl = kd_pooling_2d(ynl, pool_size=pool_size, pool_mode='avg')
                yl = kd_pooling_2d(y[l], pool_size=pool_size, pool_mode='avg')
                z[l] = tf.divide(yl, ynl)
            else:
                ynl = tf.reduce_sum(ynl, axis=1, keepdims=False)
                yl = tf.reduce_sum(yl, axis=1, keepdims=False)
                z[l] = tf.divide(yl, ynl)
    return z

def extract_samples_slices(num_points_total, num_points):
    print(num_points_total)
    assert isPowerOfTwo(num_points_total + 1)
    n = int((num_points_total + 1)/2)
    k = []
    for i in range(len(num_points)):
        assert isPowerOfTwo(num_points[i])
        m = num_points[i]
        ki = int(np.log(n / m) / np.log(2) + 0.00001)
        ki = int(n * (2**(ki+1) - 1) / (2**ki) + 0.00001)
        k.append(ki)
    return k
"""
class KdTreePooling(tf.keras.layers.Layer):
    def __init__(self, mode='MAX'):
        super(KdTreePooling, self).__init__()
        self.mode = mode

    def build(self, input_shape):
        super(KdTreePooling, self).build(input_shape)

    def call(self, x):
        return grid_pooling(x[0], x[1], x[2], x[3], x[4], mode='avg')
"""

def diameter(x, axis=-2, keepdims=True):
    return tf.reduce_max(x, axis=axis, keepdims=keepdims) - tf.reduce_min(x, axis=axis, keepdims=keepdims)

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

def kdtree_indexing_(x, depth=None):
    num_points = x.shape[1]
    #assert isPowerOfTwo(num_points)
    if depth is None:
        depth = int(np.log(num_points) / np.log(2.) + 0.1)
    y = x
    batch_idx = tf.range(x.shape[0],dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, (-1, 1))
    batch_idx = tf.tile(batch_idx, (1, x.shape[1]))

    points_idx = tf.range(num_points)
    points_idx = tf.reshape(points_idx, (1, -1, 1))
    points_idx = tf.tile(points_idx, (x.shape[0], 1, 1))



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
        points_idx = tf.gather_nd(points_idx, sort_idx)
        points_idx = tf.reshape(points_idx, (-1, int(y.shape[1] // 2), 1))
        y = tf.gather_nd(y, sort_idx)
        y = tf.reshape(y, (-1, int(y.shape[1] // 2), 3))

    y = tf.reshape(y, x.shape)
    points_idx = tf.reshape(points_idx, (x.shape[0], x.shape[1]))
    points_idx_inv = tf.argsort(points_idx, axis=-1)
    points_idx = tf.stack([batch_idx, points_idx], axis=-1)
    points_idx_inv = tf.stack([batch_idx, points_idx_inv], axis=-1)
    return y, points_idx, points_idx_inv

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

if __name__=="__main__":

    x = tf.random.uniform((2, 1024, 3))
    filename = "/home/rahul/research/data/sapien_processed/train_refrigerator.h5"
    f = h5py.File(filename, "r")
    x = f["data"][:2]
    y, kd, kd_2 = kdtree_indexing_(x)

    print(x, x.shape, y, y.shape)

    print(kd, kd.shape)
    print(kd_2, kd_2.shape)
