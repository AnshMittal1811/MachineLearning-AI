from tensorflow.python.keras.layers import Layer
import tensorflow as tf

class BuildPatches(Layer):

    def __init__(self, patch_size, centering='root', spacing=0, **kwargs):
        self.patch_size = patch_size
        self.spacing = spacing
        self.centering = centering
        super(BuildPatches, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BuildPatches, self).build(input_shape)

    def call(self, x):

        assert isinstance(x, list)
        points_pl = x[0]
        roots = x[1]

        batch_size = points_pl.get_shape()[0].value
        num_of_points = points_pl.get_shape()[1].value
        num_of_roots = roots.get_shape()[1].value
        assert(num_of_points >= self.patch_size)
        num_samples = self.patch_size

        # compute distance mat

        r0 = tf.multiply(roots, roots)
        r0 = tf.reduce_sum(r0, axis=2, keepdims=True)

        r1 = tf.multiply(points_pl, points_pl)
        r1 = tf.reduce_sum(r1, axis=2, keepdims=True)
        r1 = tf.transpose(r1, [0, 2, 1])

        sq_distance_mat = r0 - 2.*tf.matmul(roots, tf.transpose(points_pl, [0, 2, 1])) + r1


        """
        r = tf.multiply(points_pl, points_pl)
        r = tf.reduce_sum(r, 2)
        r = tf.expand_dims(r, dim=2)
        sq_distance_mat = r - 2. * tf.matmul(points_pl, tf.transpose(points_pl, [0, 2, 1])) + tf.transpose(
            r, [0, 2, 1])
        """
        # compute patches
        assert(num_samples*(self.spacing+1) <= num_of_points)

        sq_patches_dist, patches_idx = tf.nn.top_k(-sq_distance_mat, k=num_samples*(self.spacing+1))
        sq_patches_dist = -sq_patches_dist

        if self.spacing > 0:
            sq_patches_dist = sq_patches_dist[:, :, 0:(self.spacing+1):-1, ...]
            patches_idx = patches_idx[:, :, 0:(self.spacing + 1):-1, ...]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        batch_idx = tf.tile(batch_idx, (1, num_of_roots, num_samples))
        patches_idx = tf.stack([batch_idx, patches_idx], -1)

        patches = tf.gather_nd(points_pl, patches_idx)

        if self.centering == 'root':
            patches = tf.subtract(patches, tf.expand_dims(roots, axis=2))

        if self.centering == 'barycenter':
            # subtract the barycenter instead of patch center
            patch_mean = tf.reduce_mean(patches, axis=2, keepdims=True)
            patches = tf.subtract(patches, patch_mean)

        return [patches, patches_idx, sq_patches_dist, sq_distance_mat]

    def compute_output_shape(self, input_shape):
        nb = input_shape[0][0]
        nv = input_shape[1][1]
        output_shapes = []
        output_shapes.append((nb, nv, self.patch_size, 3))
        output_shapes.append((nb, nv, self.patch_size, 2))
        output_shapes.append((nb, nv, self.patch_size))
        output_shapes.append((nb, nv, nv))

        return output_shapes