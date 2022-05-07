import tensorflow as tf
import numpy as np
from SO3_CNN.sampling import tf_S2_fps, SO3_fps, SO3_sampling_from_S2, tf_polyhedrons, tf_sq_distance_matrix
from SO3_CNN.spherical_harmonics_ import tf_spherical_harmonics
import vispy


def max_dirac_regularizer(x, phi_values, bound=True):
    X = []
    Phi = []
    for l in x:
        if l.isnumeric() and l in phi_values:
            Phi.append(phi_values[l])
            X.append(x[l])

    Phi = tf.concat(Phi, axis=-1)
    X = tf.concat(X, axis=-2)
    eval = tf.einsum('ij,...jc->...ic', Phi, X)
    eval = tf.sort(eval, axis=-2)
    shape = list(eval.shape)
    shape[-2] = 1
    o = tf.ones(shape)
    shape[-2] = eval.shape[-2] - 1
    z = tf.zeros(shape)
    dirac = tf.concat([z, o], axis=-2)
    if not bound:
        dirac = tf.multiply(dirac, eval)
    y = tf.subtract(eval, dirac)
    y = tf.multiply(y, y)
    y = tf.reduce_mean(y)
    return y

def unimodality_regularizer(x, phi_values, nn_idx):

    X = []
    Phi = []
    for l in x:
        if l.isnumeric() and l in phi_values:
            Phi.append(phi_values[l])
            X.append(x[l])

    Phi = tf.concat(Phi, axis=-1)
    X = tf.concat(X, axis=-2)
    eval = tf.einsum('ij,...jc->...ic', Phi, X)
    shape = list(eval.shape)
    shape.insert(-2, nn_idx.shape[-1])
    # patches = tf.gather_nd(eval, nn_idx)
    patches = tf.gather(eval, nn_idx, axis=-2)
    # patches = tf.reshape(patches, shape)
    max_patches = tf.reduce_max(patches, axis=-2, keepdims=True)

    # sorted_patches = tf.sort(patches, axis=-2)
    # second_to_highest = sorted_patches[..., -2, :]

    global_max = tf.reduce_max(eval, axis=-2, keepdims=True)

    is_global_max = tf.greater_equal(eval, global_max)


    is_local_max = tf.greater_equal(eval, max_patches)
    is_non_global_max = tf.logical_and(is_local_max, tf.logical_not(is_global_max))
    is_negative = tf.greater(-eval, 0.)




    mask = tf.math.logical_or(tf.logical_not(is_non_global_max), is_negative)
    # mask = tf.logical_or(mask, is_negative)


    # mask = is_global_max

    mask = tf.cast(mask, tf.float32)

    normalized_eval = tf.divide(eval, tf.maximum(global_max, 0.00001))
    # mass = tf.reduce_sum(tf.abs(eval), axis=-2, keepdims=True)
    # normalized_eval = tf.divide(eval, tf.maximum(mass, 0.00001))


    y = tf.subtract(eval, tf.multiply(normalized_eval, mask))
    y = tf.multiply(y, y)
    # y = tf.divide(tf.reduce_sum(y), tf.reduce_sum(mask) + 0.001)
    y = tf.reduce_mean(y)
    return y

class S2UnimodalityRegularizer:
    def __init__(self, num_samples, l_max=3, l_list=None, sph_fn=None, num_neighbours=None, regularizer='max_dirac'):
        if sph_fn is not None:
            if l_list is not None:
                self.l_list = l_list
                self.l_max = max(l_list)
            else:
                self.l_list = range(l_max+1)
                self.l_max = l_max
            self.sph_fn = sph_fn
        else:
            self.sph_fn = tf_spherical_harmonics(l_max=l_max, l_list=l_list)

        self.num_samples = num_samples
        self.num_neighbours = num_neighbours
        self.regularizer = regularizer
        self.S2 = tf_S2_fps(num_samples, res=max(100, int(20*np.sqrt(num_samples))))

        self.Y = self.sph_fn.compute(self.S2)
        """
        self.types = y.keys()
        Y = []
        for l in self.types:
            Y.append(tf.reshape(y[l], (-1, 2 * int(l) + 1)))
        self.Y = tf.concat(Y, axis=-1)
        """
        if regularizer != 'max_dirac':
            self.num_neighbours = 6

        self.nn_idx = None
        if self.num_neighbours is not None:
            self.num_neighbours = num_neighbours
            sq_dist_mat = tf_sq_distance_matrix(self.S2, self.S2)
            _, patches_idx = tf.nn.top_k(-sq_dist_mat, k=self.num_neighbours)
            self.nn_idx = patches_idx

    def loss(self, x):
        if self.regularizer is 'max_dirac':
            return max_dirac_regularizer(x, self.Y, bound=True)
        else:
            return unimodality_regularizer(x, self.Y, self.nn_idx)

