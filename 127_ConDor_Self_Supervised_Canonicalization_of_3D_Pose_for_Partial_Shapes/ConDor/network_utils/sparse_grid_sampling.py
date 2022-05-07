import tensorflow as tf

def grid_sampler(x, cell_size, offset=64, num_points_target=None):
    points = x
    batch_size = points.shape[0]
    num_points_source = points.shape[1]

    # points = tf.reshape(points, (-1, 3))
    rounded_points = tf.round(points / cell_size)
    grid_idx = tf.cast(rounded_points, dtype=tf.int32)
    rounded_points = cell_size*rounded_points

    min_grid_idx = tf.reduce_min(grid_idx)
    grid_size = tf.reduce_max(grid_idx) - min_grid_idx + 2*offset
    grid_idx = grid_idx - min_grid_idx + offset
    # batch_idx_offset = (grid_size**3)*batch_idx
    linear_idx = grid_size * (grid_size * grid_idx[..., 0] + grid_idx[..., 1]) + grid_idx[..., 2]

    # max_linear_idx_batch = tf.math.unsorted_segment_max(linear_idx, batch_idx_in, batch_size)
    max_linear_idx_batch = tf.reduce_max(linear_idx, axis=1, keepdims=False)

    batch_linear_idx_offset = tf.math.cumsum(2*max_linear_idx_batch, exclusive=True)
    # batch_linear_idx_offset = tf.gather(batch_linear_idx_offset, batch_idx_in, axis=0)
    batch_linear_idx_offset = tf.expand_dims(batch_linear_idx_offset, axis=1)
    linear_idx = tf.add(batch_linear_idx_offset, linear_idx)
    linear_idx = tf.reshape(linear_idx, (-1,))
    unique_linear_idx, cell_idx = tf.unique(linear_idx)

    cell_idx_inv = tf.math.unsorted_segment_min(tf.range(cell_idx.shape[0])+1, cell_idx, cell_idx.shape[0])-1
    cell_idx_inv = tf.reshape(cell_idx_inv, (batch_size, num_points_source))
    sort_idx = tf.argsort(cell_idx_inv, axis=1, direction='DESCENDING')

    batch_idx = tf.expand_dims(tf.range(batch_size), axis=1)
    batch_idx = tf.tile(batch_idx, (1, num_points_source))
    idx = tf.stack([batch_idx, sort_idx], axis=-1)
    rounded_points = tf.gather_nd(rounded_points, idx)
    rounded_points = rounded_points[:, :num_points_target, :]
    """
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(unique_linear_idx, tf.range(unique_linear_idx.shape[0])), -1)


    # unique_linear_idx, cell_idx = tf.unique(linear_idx)
    # rounded_points = tf.gather(rounded_points, cell_idx_inv, axis=0)

    num_cells = unique_linear_idx.shape[0]
    batch_idx_out = tf.math.unsorted_segment_min(batch_idx_in, cell_idx, num_segments=num_cells)


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
    """
    return rounded_points