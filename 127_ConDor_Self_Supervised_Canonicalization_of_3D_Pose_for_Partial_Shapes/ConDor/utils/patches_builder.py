import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from convolution.pooling import UniformSampling

def normalize_patches_(patches, patches_dist):
    patches_radius = tf.reduce_max(patches_dist, axis=-1, keepdims=True)
    patches_radius = tf.reduce_mean(patches_radius, axis=1, keepdims=True)
    patches_dist = tf.divide(patches_dist, patches_radius)
    patches_radius = tf.expand_dims(patches_radius, axis=-1)
    patches = tf.divide(patches, patches_radius)
    return patches, patches_dist

def compute_patches_(points, roots,
                    sq_distance_mat,
                    batch_size,
                    num_of_points,
                    num_of_roots,
                    num_samples,
                    spacing,
                    normalize_patches):
    assert (num_samples * (spacing + 1) <= num_of_points)
    sq_patches_dist, patches_idx = tf.nn.top_k(-sq_distance_mat, k=num_samples * (spacing + 1))
    sq_patches_dist = -sq_patches_dist
    if spacing > 0:
        sq_patches_dist = sq_patches_dist[:, :, 0:(spacing + 1):-1, ...]
        patches_idx = patches_idx[:, :, 0:(spacing + 1):-1, ...]
    patches_dist = tf.sqrt(tf.maximum(sq_patches_dist, 0.00001))
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, num_of_roots, num_samples))
    patches_idx = tf.stack([batch_idx, patches_idx], -1)
    patches = tf.gather_nd(points, patches_idx)
    # if self.centering == 'root':
    patches = tf.subtract(patches, tf.expand_dims(roots, axis=2))
    if normalize_patches:
        patches, patches_dist = normalize_patches_(patches, patches_dist)
    return patches, patches_idx, patches_dist

def pc_hierarchy(points_input, params):
    assert ('num_points' in params)
    num_points = params['num_points']
    if 'pooling' in params:
        Pooling = params['pooling']
    else:
        Pooling = UniformSampling
    points_input = tf.convert_to_tensor(points_input, dtype=tf.float32)
    points_input = Pooling(ratio=num_points[0] / points_input.get_shape()[1])(points_input)
    points = [points_input]
    for i in range(len(num_points) - 1):
        points.append(Pooling(ratio=num_points[i + 1] / num_points[i])(points[i]))
    return points

def patches_hierarchy(points, params):
    # assert('num_points' in params)
    # num_points = params['num_points']
    assert('patch_size' in params)
    patch_size = params['patch_size']
    if 't_patch_size' in params:
        if params['t_patch_size'] is None:
            t_patch_size = patch_size
        else:
            t_patch_size = params['t_patch_size']
    else:
        t_patch_size = patch_size
    if 'transposed_patches' in params:
        transposed_patches = params['transposed_patches']
    else:
        transposed_patches = False
    if 'normalize_patches' in params:
        normalize_patches = params['normalize_patches']
    else:
        normalize_patches = False
    if 'spacing' in params:
        spacing = params['spacing']
    else:
        spacing = 0
    if 'return_patch_dist' in params:
        return_patch_dist = params['return_patch_dist']
    else:
        return_patch_dist = False

    # points = []
    patches = []
    patches_idx = []
    patches_dist = []
    t_patches = []
    t_patches_idx = []
    t_patches_dist = []
    sq_dist = []
    """
    for i in range(len(num_points)):
        points.append(Input(batch_shape=(batch_size, num_points[i], 3)))
    """
    for i in range(len(points)-1):
        P = BuildPatches(patch_size=patch_size[i],
                         t_patch_size=t_patch_size[i],
                         centering='root',
                         spacing=spacing,
                         normalize_patches=normalize_patches,
                         transposed_patches=False)([points[i], points[i+1]])
        if transposed_patches:
            # [patches, t_patches, patches_idx, t_patches_idx, patches_dist, t_patches_dist, sq_distance_mat]
            patches.append(P[0])
            t_patches.append(P[1])
            patches_idx.append(P[2])
            t_patches_idx.append(P[3])
            patches_dist.append(P[4])
            t_patches_dist.append(P[5])
            sq_dist.append(P[-1])
        else:
            patches.append(P[0])
            patches_idx.append(P[1])
            patches_dist.append(P[2])
            sq_dist.append(P[-1])

    if transposed_patches:
        return patches, t_patches, patches_idx, t_patches_idx, patches_dist, t_patches_dist, sq_dist
    else:
        return patches, patches_idx, patches_dist, sq_dist


class BuildPatches(Layer):
    def __init__(self, patch_size, t_patch_size=None, centering='root', spacing=0, normalize_patches=False,
                 transposed_patches=False, **kwargs):
        self.patch_size = patch_size
        self.spacing = spacing
        self.centering = centering
        self.normalize_patches = normalize_patches
        self.transposed_patches = transposed_patches
        if t_patch_size is None:
            self.t_patch_size = patch_size
        else:
            self.t_patch_size = t_patch_size
        super(BuildPatches, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BuildPatches, self).build(input_shape)

    def call(self, x):

        assert isinstance(x, list)
        points = x[0]
        roots = x[1]

        batch_size = points.get_shape()[0]
        num_of_points = points.get_shape()[1]
        num_of_roots = roots.get_shape()[1]
        assert(num_of_points >= self.patch_size)

        # compute distance mat

        r0 = tf.multiply(roots, roots)
        r0 = tf.reduce_sum(r0, axis=2, keepdims=True)

        r1 = tf.multiply(points, points)
        r1 = tf.reduce_sum(r1, axis=2, keepdims=True)
        r1 = tf.transpose(r1, [0, 2, 1])

        sq_distance_mat = r0 - 2.*tf.matmul(roots, tf.transpose(points, [0, 2, 1])) + r1


        """
        r = tf.multiply(points_pl, points_pl)
        r = tf.reduce_sum(r, 2)
        r = tf.expand_dims(r, dim=2)
        sq_distance_mat = r - 2. * tf.matmul(points_pl, tf.transpose(points_pl, [0, 2, 1])) + tf.transpose(
            r, [0, 2, 1])
        """
        # compute patches

        patches, patches_idx, patches_dist = compute_patches_(points=points,
                                                              roots=roots,
                                                              sq_distance_mat=sq_distance_mat,
                                                              batch_size=batch_size,
                                                              num_of_points=num_of_points,
                                                              num_of_roots=num_of_roots,
                                                              num_samples=self.patch_size,
                                                              spacing=self.spacing,
                                                              normalize_patches=self.normalize_patches)

        """
        if self.centering == 'barycenter':
            # subtract the barycenter instead of patch center
            patch_mean = tf.reduce_mean(patches, axis=2, keepdims=True)
            patches = tf.subtract(patches, patch_mean)
        """
        if not self.transposed_patches:
            return [patches, patches_idx, patches_dist, sq_distance_mat]
        else:
            t_sq_distance_mat = tf.transpose(sq_distance_mat, [0, 2, 1])
            t_patches, t_patches_idx, t_patches_dist = compute_patches_(points=roots,
                                                                        roots=points,
                                                                        sq_distance_mat=t_sq_distance_mat,
                                                                        batch_size=batch_size,
                                                                        num_of_points=num_of_roots,
                                                                        num_of_roots=num_of_points,
                                                                        num_samples=self.t_patch_size,
                                                                        spacing=self.spacing,
                                                                        normalize_patches=self.normalize_patches)
            return [patches, t_patches, patches_idx, t_patches_idx, patches_dist, t_patches_dist, sq_distance_mat]
            # return [patches, patches_idx, patches_dist, patches, patches_idx, patches_dist, sq_distance_mat]

    def compute_output_shape(self, input_shape):
        nb = input_shape[0][0]
        nr = input_shape[1][1]
        nv = input_shape[0][1]
        if not self.transposed_patches:
            output_shapes = []
            output_shapes.append((nb, nr, self.patch_size, 3))
            output_shapes.append((nb, nr, self.patch_size, 2))
            output_shapes.append((nb, nr, self.patch_size))
            output_shapes.append((nb, nr, nv))
        else:
            output_shapes = []
            output_shapes.append((nb, nr, self.patch_size, 3))
            output_shapes.append((nb, nv, self.t_patch_size, 3))
            output_shapes.append((nb, nr, self.patch_size, 2))
            output_shapes.append((nb, nv, self.t_patch_size, 2))
            output_shapes.append((nb, nr, self.patch_size))
            output_shapes.append((nb, nv, self.t_patch_size))
            output_shapes.append((nb, nr, nv))
        return output_shapes