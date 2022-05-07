import tensorflow as tf
import numpy as np

def tf_unique_with_inverse(x):
    y, idx = tf.unique(x)
    num_segments = tf.shape(y)[0]
    num_elems = tf.shape(x)[0]
    return (y, idx,  tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments))

def compute_patches_grid(source, target, num_samples, radius):
    batch_size = source.shape[0]
    num_points_source = source.shape[1]
    num_points_target = target.shape[1]
    k = int(np.ceil(pow(num_samples, 1/2.5))) # int(np.ceil(pow(num_samples, 1/3.)))
    cell_size = radius / k
    source_round = tf.round(source / cell_size)
    target_round = tf.round(source / cell_size)
    source_grid_idx = tf.cast(source_round, dtype=tf.int32)
    target_grid_idx = tf.cast(target_round, dtype=tf.int32)
    min_grid_idx = tf.minimum(tf.reduce_min(source_grid_idx), tf.reduce_min(target_grid_idx))
    max_grid_idx = tf.maximum(tf.reduce_max(source_grid_idx), tf.reduce_max(target_grid_idx))
    source_grid_idx = source_grid_idx - min_grid_idx + k
    target_grid_idx = target_grid_idx - min_grid_idx + k
    grid_size = max_grid_idx - min_grid_idx + 2*(k + 1)
    source_round = cell_size*source_round
    target_round = cell_size*target_round
    source_linear_idx = grid_size * (grid_size * source_grid_idx[0] + source_grid_idx[1]) + source_grid_idx[2]
    target_linear_idx = grid_size * (grid_size * target_grid_idx[0] + target_grid_idx[1]) + target_grid_idx[2]
    batch_idx = tf.expand_dims(tf.multiply(grid_size ** 3, tf.range(batch_size)), axis=-1)
    source_linear_idx = tf.add(batch_idx, source_linear_idx)
    target_linear_idx = tf.add(batch_idx, target_linear_idx)
    target_linear_idx = tf.reshape(target_linear_idx, [-1])
    target_ul, target_ul_idx, target_ul_idx_inv = tf_unique_with_inverse(target_linear_idx)
    s = tf.range(-k, k+1)
    shift = tf.stack(tf.meshgrid(s, s, s), axis=-1)
    shift_norm2 = tf.reshape(tf.reduce_sum(tf.multiply(shift, shift), axis=-1, keepdims=False), [-1])
    num_cells = tf.reduce_sum(shift_norm2 <= radius*radius)
    # _, top_k_idx = tf.math.top_k(-shift_norm2, k=int(0.6*(2*k+1)**3))
    _, top_k_idx = tf.math.top_k(-shift_norm2, k=num_cells)
    shift_linear_idx = tf.reshape(grid_size * (grid_size * shift[0] + shift[1]) + shift[2], [1, 1, -1])
    shift_linear_idx = tf.gather(shift_linear_idx, top_k_idx, axis=-1)
    source_linear_idx = tf.tile(tf.reshape(source_linear_idx, [batch_size, num_points_source, 1]), [1, 1, num_cells])
    source_linear_idx = tf.add(shift_linear_idx, source_linear_idx)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(target_ul, target_ul_idx_inv), -1)
    patches_idx = table.lookup(source_linear_idx)
    patches_idx = tf.reshape(patches_idx, [batch_size, num_points_source, -1])
    target = target + tf.reduce_min(target) + 10*radius
    patches = tf.gather(tf.reshape(target, [-1, 3]), patches_idx, axis=0)
    patches = tf.subtract(patches, tf.expand_dims(target, axis=-2))
    patches_norm2 = tf.reduce_sum(tf.multiply(patches, patches), axis=-1, keepdims=False)
    neg_patches_norm2, top_k_idx = tf.math.top_k(-patches_norm2, k=num_samples)
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, num_points_source, num_samples))
    point_idx = tf.range(0, num_points_source)
    point_idx = tf.tile(point_idx, (batch_size, 1, num_samples))
    top_k_idx = tf.stack([batch_idx, point_idx, top_k_idx])
    patches_idx = tf.gather_nd(patches_idx, top_k_idx)
    cond = tf.less_equal(-radius, neg_patches_norm2)
    num_point_patches = tf.cast(tf.reduce_sum(cond, axis=-1, keepdims=False), tf.float32)
    patches_idx = tf.where(cond, patches_idx, -1)
    patches = tf.gather_nd(patches, top_k_idx)
    return patches, patches_idx, num_point_patches

def patches_radius(radius, sq_norm):
    batch_size = sq_norm.shape[0]
    rad = radius
    if isinstance(radius, float):
        rad = radius * tf.ones((batch_size, 1, 1))
    if isinstance(radius, str):
        rad = tf.sqrt(tf.maximum(tf.reduce_max(sq_norm, axis=2, keepdims=False), 0.0000001))
        if radius == "avg":
            rad = tf.reduce_mean(rad, axis=-1, keepdims=False)
        elif radius == 'min':
            rad = tf.reduce_min(rad, axis=-1, keepdims=False)
        elif radius.isnumeric():
            rad = tf.sort(rad, axis=-1)
            i = int((float(int(radius)) / 100.) * sq_norm.shape[1])
            i = max(i, 1)
            rad = tf.reduce_mean(rad[:, :i], axis=-1, keepdims=False)
        rad = tf.reshape(rad, (batch_size, 1, 1))
    return rad

def compute_patches_(source, target, sq_distance_mat, num_samples, spacing, radius, source_mask=None):
    batch_size = source.shape[0]
    num_points_source = source.shape[1]
    num_points_target = target.shape[1]
    assert (num_samples * (spacing + 1) <= num_points_source)

    sq_patches_dist, patches_idx = tf.nn.top_k(-sq_distance_mat, k=num_samples * (spacing + 1))
    sq_patches_dist = -sq_patches_dist
    if spacing > 0:
        sq_patches_dist = sq_patches_dist[:, :, 0::(spacing + 1), ...]
        patches_idx = patches_idx[:, :, 0::(spacing + 1), ...]



    rad = patches_radius(radius, sq_patches_dist)




    patches_size = patches_idx.shape[-1]

    # mask = sq_patches_dist < radius ** 2
    mask = tf.greater_equal(rad ** 2, sq_patches_dist)
    patches_idx = tf.cast(tf.where(mask, patches_idx, -1), dtype=tf.int32)
    if source_mask is not None:
        source_mask = source_mask < 1
        source_mask = tf.tile(tf.expand_dims(source_mask, axis=-1), (1, 1, patches_idx.shape[-1]))
        patches_idx = tf.where(source_mask, patches_idx, -1)

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, num_points_target, num_samples))
    patches_idx = tf.stack([batch_idx, patches_idx], -1)

    source = tf.divide(source, rad)
    target = tf.divide(target, rad)
    patches = tf.gather_nd(source, patches_idx)
    # print(source, patches)
    patches = tf.subtract(patches, tf.expand_dims(target, axis=-2))
    # print("target", target)

    if source_mask is not None:
        mask = source_mask
    else:
        mask = tf.ones((batch_size, num_points_source))
    # print(mask, "mask")
    patch_size = tf.gather_nd(mask, patches_idx)
    patches_size = tf.reduce_sum(patch_size, axis=-1, keepdims=False)
    patches_dist = tf.sqrt(tf.maximum(sq_patches_dist, 0.000000001))
    patches_dist = tf.divide(patches_dist, rad)
    return {"patches": patches, "patches idx": patches_idx, "patches size": patches_size, "patches radius": rad,
            "patches dist": patches_dist}

class GroupPoints(tf.keras.layers.Layer):
    def __init__(self, radius, patch_size_source, radius_target=None, patch_size_target=None,
                 spacing_source=0, spacing_target=0):
        super(GroupPoints, self).__init__()
        self.radius = radius
        self.radius_target = radius_target
        self.patch_size_source = patch_size_source
        self.patch_size_target = patch_size_target
        self.spacing_source = spacing_source
        self.spacing_target = spacing_target
    def build(self, input_shape):
        super(GroupPoints, self).build(input_shape)

    def call(self, x):
        """
        :param x: [source, target]
        :return: [patches_idx_source, num_incident_points_target]
        """
        assert isinstance(x, dict)
        source = x["source points"]
        target = x["target points"]

        source_mask = None
        if "source mask" in x:
            source_mask = x["source mask"]

        target_mask = None
        if "target mask" in x:
            target_mask = x["target mask"]

        num_points_source = source.shape[1]

        # assert (num_points_source >= self.patch_size_source)
        if self.patch_size_target is not None:
            num_points_target = target.shape[1]
            # assert (num_points_target >= self.patch_size_source)

        # compute distance mat
        r0 = tf.multiply(target, target)
        r0 = tf.reduce_sum(r0, axis=2, keepdims=True)
        r1 = tf.multiply(source, source)
        r1 = tf.reduce_sum(r1, axis=2, keepdims=True)
        r1 = tf.transpose(r1, [0, 2, 1])
        sq_distance_mat = r0 - 2. * tf.matmul(target, tf.transpose(source, [0, 2, 1])) + r1


        patches = compute_patches_(source, target, sq_distance_mat,
                                   min(self.patch_size_source, num_points_source),
                                   self.spacing_source, self.radius,
                                   source_mask=source_mask)
        y = dict()
        y["patches source"] = patches["patches"]
        y["patches idx source"] = patches["patches idx"]
        y["patches size source"] = patches["patches size"]
        y["patches radius source"] = patches["patches radius"]
        y["patches dist source"] = patches["patches dist"]

        # y = [patches_source, patches_idx_source, patches_size_source]
        if self.patch_size_target is not None:
            sq_distance_mat_t = tf.transpose(sq_distance_mat, perm=(0, 2, 1))
            patches = compute_patches_(target, source, sq_distance_mat_t,
                                       min(self.patch_size_target, num_points_target),
                                       self.spacing_target, self.radius_target,
                                       source_mask=target_mask)
            # y += [patches_target, patches_idx_target, patches_size_target]

            y["patches target"] = patches["patches"]
            y["patches idx target"] = patches["patches idx"]
            y["patches size target"] = patches["patches size"]
            y["patches radius target"] = patches["patches radius"]
            y["patches dist target"] = patches["patches dist"]
        # y.append(radius)

        return y

if __name__ == "__main__":

    gi = GroupPoints(0.2, 32)
    N_pts = 10
    start = 10
    x = tf.ones((2, N_pts, 3)) * tf.expand_dims(tf.expand_dims(tf.range(N_pts, dtype=tf.float32), -1), 0)
    y = tf.ones((2, N_pts, 3)) * tf.expand_dims(tf.expand_dims(tf.range(start, N_pts + start, dtype=tf.float32), -1), 0)
    # print(x, y)
    out = gi({"source points": x, "target points": y})
    for k in out:
        print(out[k], out[k].shape, " ", k)