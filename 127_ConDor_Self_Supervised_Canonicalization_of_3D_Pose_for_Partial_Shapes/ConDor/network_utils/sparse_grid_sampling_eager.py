import tensorflow as tf
import math
from tensorflow.python.keras.layers import AveragePooling2D, AveragePooling1D, MaxPooling1D, MaxPooling2D

"""
computes unique elements y in a 1D tensor x
and idx such that x[i] = y[idx[i]]
and inverse indices such that idx[idx_inv[j]] = j
"""
def tf_unique_with_inverse(x):
    print(x)
    y, idx = tf.unique(x)
    # y, idx, counts = tf.unique_with_counts(x)
    num_segments = y.shape[0]
    num_elems = x.shape[0]
    return (y, idx,  tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments))

def crop_points(batch_idx, limit):
    batch_sort_idx = tf.argsort(batch_idx, axis=0)
    batch_idx_ = tf.gather(batch_idx, batch_sort_idx, axis=0)
    num_points = tf.math.segment_sum(tf.ones(batch_idx_.shape), batch_idx_)
    num_points_cropped = tf.minimum(num_points, limit)
    num_removed_points = num_points - num_points_cropped
    cum_removed_points = tf.cumsum(num_removed_points, exclusive=True)
    cum_removed_points = tf.repeat(cum_removed_points, tf.cast(num_points_cropped, dtype=tf.int32), axis=0)
    idx = tf.range(tf.reduce_sum(num_points_cropped), dtype=tf.float32)
    idx = tf.cast(tf.add(idx, cum_removed_points), tf.int32)
    idx = tf.gather(batch_sort_idx, idx, axis=0)

    """
    # idx = batch_sort_idx
    idx = tf.split(batch_sort_idx, num_or_size_splits=num_points)
    I = []

    for i in range(batch_size):
        limit_i = tf.cast(tf.minimum(limit, num_points[i]), tf.int32)
        I.append(idx[i][:limit_i])

    idx = tf.concat(I, axis=0)
    """

    """
    # num_points_cropped =
    # batch_size = num_points.shape[0]
    cum_num_points = tf.cast(tf.cumsum(num_points, exclusive=True), tf.int32)
    
    
    idx_i = tf.range(limit, dtype=tf.int32)
    idx_i = tf.random.shuffle(idx_i)
    idx = []
    cumsum = 0
    for i in range(batch_size):
        # print(limit)
        # print(tf.minimum(limit, num_points[i]))
        limit_i = tf.cast(tf.minimum(limit, num_points[i]), tf.int32)

        idx.append(tf.add(idx_i[:limit_i], cum_num_points[i]))
        cumsum += limit_i
    idx = tf.concat(idx, axis=0)
    idx = tf.gather(batch_sort_idx, idx, axis=0)
    return idx
    """
    return idx

def grid_sampler(x, cell_size, offset=64, num_points_target=None, batch_size=None):
    points = x["points"]
    batch_idx_in = x["batch idx"]
    if batch_size is None:
        batch_size = tf.reduce_max(batch_idx_in) + 1
    rounded_points = tf.round(points / cell_size)
    grid_idx = tf.cast(rounded_points, dtype=tf.int32)
    rounded_points = cell_size*rounded_points
    min_grid_idx = tf.reduce_min(grid_idx)
    grid_size = tf.reduce_max(grid_idx) - min_grid_idx + 2*offset
    grid_idx = grid_idx - min_grid_idx + offset
    # batch_idx_offset = (grid_size**3)*batch_idx
    linear_idx = grid_size * (grid_size * grid_idx[..., 0] + grid_idx[..., 1]) + grid_idx[..., 2]

    print(batch_idx_in.shape)
    print(linear_idx.shape)
    max_linear_idx_batch = tf.math.unsorted_segment_max(linear_idx, batch_idx_in, batch_size)

    batch_linear_idx_offset = tf.math.cumsum(2*max_linear_idx_batch, exclusive=True)
    batch_linear_idx_offset = tf.gather(batch_linear_idx_offset, batch_idx_in, axis=0)
    linear_idx = tf.add(batch_linear_idx_offset, linear_idx)
    unique_linear_idx, cell_idx, cell_idx_inv = tf_unique_with_inverse(linear_idx)
    # unique_linear_idx, cell_idx = tf.unique(linear_idx)
    # rounded_points = tf.gather(rounded_points, cell_idx_inv, axis=0)

    num_cells = unique_linear_idx.shape[0]
    batch_idx_out = tf.math.unsorted_segment_min(batch_idx_in, cell_idx, num_segments=num_cells)


    """
    if num_points_target is not None:
        points_idx = tf.random.shuffle(tf.range(num_cells))
        points_idx_inv = tf.argsort(points_idx)
        cell_idx = tf.gather(points_idx_inv, cell_idx, axis=0)
        num_cells = tf.minimum(num_cells, batch_size*num_points_target)
        points_idx = points_idx[:num_cells]
        batch_idx_out = tf.gather(batch_idx_out, points_idx)
        unique_linear_idx = tf.gather(unique_linear_idx, points_idx, axis=0)
        rounded_points = tf.gather(rounded_points, points_idx, axis=0)
    """

    if num_points_target is not None:
        idx = crop_points(batch_idx_out, num_points_target)
        num_cells = idx.shape[0]

        # idx_inv = tf.argsort(idx, axis=0) + 1
        # cell_idx = tf.gather(idx_inv, cell_idx, axis=0) - 1

        table_ = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(idx, tf.range(idx.shape[0])), -1)
        cell_idx = table_.lookup(cell_idx)

        u = tf.stack([batch_idx_out, unique_linear_idx, cell_idx_inv], axis=0)
        u = tf.gather(u, idx, axis=-1)
        batch_idx_out = u[0]
        unique_linear_idx = u[1]
        cell_idx_inv = u[2]


    rounded_points = tf.gather(rounded_points, cell_idx_inv, axis=0)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(unique_linear_idx, tf.range(unique_linear_idx.shape[0])), -1)


    y = dict()
    y["batch idx"] = batch_idx_out
    y["linear idx"] = unique_linear_idx
    y["lookup table"] = table
    y["cell idx"] = cell_idx
    y["num cells"] = num_cells
    y["rounded points"] = rounded_points
    y["grid size"] = grid_size
    return y


def extract_batch_idx(i, batch_idx, x):
    batch_size = tf.reduce_max(batch_idx) + 1
    batch_sort_idx = tf.argsort(batch_idx, axis=0)
    batch_idx = tf.gather(batch_idx, batch_sort_idx, axis=0)
    x = tf.gather(x, batch_sort_idx, axis=0)
    num_points = tf.math.segment_sum(tf.ones(batch_idx.shape), batch_idx)
    print(num_points)
    num_points = tf.cast(tf.cumsum(num_points, exclusive=True), tf.int32)
    if i == batch_size - 1:
        return x[num_points[i]:]
    else:
        print(num_points[i])
        print(num_points[i+1])
        return x[num_points[i]:num_points[i+1]]


class GridSampler(tf.keras.layers.Layer):
    def __init__(self, cell_size, limit=None):
        super(GridSampler, self).__init__()
        self.cell_size = cell_size
        self.limit = limit

    def build(self, input_shape):
        super(GridSampler, self).build(input_shape)

    def call(self, x):
        return grid_sampler(x, self.cell_size, offset=64,
                            num_points_target=self.limit, batch_size=None)

def grid_sampling(x, cell_size, limit, batch_size=None, num_points=None):
    if batch_size is None:
        batch_size = x.shape[0]
    if num_points is None:
        num_points = x.shape[1]

    batch_idx = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1)
    batch_idx = tf.tile(batch_idx, (1, num_points))
    batch_idx = tf.reshape(batch_idx, (-1,))
    x = tf.reshape(x, (-1, 3))
    y = []
    for i in range(len(cell_size)):
        yi = GridSampler(cell_size[i], limit=limit[i])({"points": x, "batch idx": batch_idx})
        y.append(yi)
        x = yi["rounded points"]
        batch_idx = yi["batch idx"]
    return y


class GlobalGridPooling(tf.keras.layers.Layer):
    def __init__(self, pool_mode, batch_size=None):
        super(GlobalGridPooling, self).__init__()
        self.pool_mode = pool_mode
        self.batch_size = batch_size

    def build(self, input_shape):
        super(GlobalGridPooling, self).build(input_shape)

    def call(self, x):
        batch_size = self.batch_size
        if isinstance(x, list):
            if self.batch_size is None:
                batch_size = tf.reduce_max(x[1]) + 1
            if self.pool_mode == 'MAX' or self.pool_mode == 'max':
                return tf.math.unsorted_segment_max(x[0], x[1], num_segments=batch_size)
            else:
                return tf.math.unsorted_segment_mean(x[0], x[1], num_segments=batch_size)
        else:
            if self.batch_size is None:
                batch_size = tf.reduce_max(x["batch idx"]) + 1
            assert(isinstance(x, dict))
            if self.pool_mode == 'MAX' or self.pool_mode == 'max':
                return tf.math.unsorted_segment_max(x["signal"], x["batch idx"], num_segments=batch_size)
            else:
                return tf.math.unsorted_segment_mean(x["signal"], x["batch idx"], num_segments=batch_size)

class GridPooling(tf.keras.layers.Layer):
    def __init__(self, pool_mode):
        super(GridPooling, self).__init__()
        self.pool_mode = pool_mode

    def build(self, input_shape):
        super(GridPooling, self).build(input_shape)

    def call(self, x):
        if isinstance(x, list):
            if self.pool_mode == 'MAX' or self.pool_mode == 'max':
                return tf.math.unsorted_segment_max(x[0], x[1], num_segments=x[2])
            else:
                return tf.math.unsorted_segment_mean(x[0], x[1], num_segments=x[2])
        else:
            assert(isinstance(x, dict))
            if self.pool_mode == 'MAX' or self.pool_mode == 'max':
                return tf.math.unsorted_segment_max(x["signal"], x["cell idx"], num_segments=x["num cells"])
            else:
                return tf.math.unsorted_segment_mean(x["signal"], x["cell idx"], num_segments=x["num cells"])




