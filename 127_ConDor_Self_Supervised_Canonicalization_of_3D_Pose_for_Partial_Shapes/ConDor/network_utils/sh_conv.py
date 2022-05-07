import tensorflow as tf
from spherical_harmonics.tf_spherical_harmonics import normalized_sh, unnormalized_sh
import numpy as np

def tf_heaviside(X):
    return tf.maximum(0.0, tf.sign(X))


def tf_segment_indicator_(X, a, b):
    return tf_heaviside(X - a) - tf_heaviside(X - b)


def tf_segment_indictor(X, r, sigma):
    return tf_segment_indicator_(X, r - sigma, r + sigma)


def tf_hat(x, sigma):
    x = x / sigma
    return 0.5*(tf.nn.relu(x + 1.) - 2. * tf.nn.relu(x) + tf.nn.relu(x - 1.))


def tf_gaussian(x, sigma):
    print('sigma !!!!', sigma)
    # sigma = 3*sigma
    x2 = tf.multiply(x, x)
    return tf.exp(-x2 / (2. * (sigma ** 2)))




"""
def tf_gaussian(x, sigma):
    # print('sigam !! ', 3.*sigma)
    return 0.*x + 1.
"""


def tf_zero(x, sigma):
    return x


def tf_sh_kernel(X, sq_dist, nr, l_max, rad, radial_weights_fn, normalize_patch=False,
                  dtype=tf.float32):
    # Y = unnormalized_sh(X, l_max, dtype=dtype)
    Y = normalized_sh(X, l_max, dtype=dtype, eps=0.0001)

    dist = tf.sqrt(tf.maximum(sq_dist, 0.0001))

    if normalize_patch:
        radius = tf.reduce_max(dist, axis=2, keepdims=True)
        radius = tf.reduce_mean(radius, axis=1, keepdims=True)
        dist = tf.divide(dist, radius + 0.0001)
        rad = 0.8

    dist = tf.expand_dims(dist, axis=-1)
    r = tf.reshape(tf.lin_space(start=0., stop=rad, num=nr), shape=(1, 1, 1, nr))
    r = tf.subtract(dist, r)
    sigma = np.sqrt(rad/(2.*nr))
    radial_weights = radial_weights_fn(r, sigma)
    rw = tf.reduce_mean(radial_weights, axis=[-2, -1], keepdims=True)
    radial_weights = tf.divide(radial_weights, tf.maximum(rw, 0.0001))
    Y = tf.expand_dims(Y, axis=-1)
    radial_weights = tf.expand_dims(radial_weights, axis=-2)
    y = tf.multiply(Y, radial_weights)

    # y = tf.expand_dims(Y, axis=-1)
    # y = tf.tile(y, multiples=(1, 1, 1, 1, 3))
    return y


class ShKernel:
    def __init__(self, nr, l_max, rad, radial_fn=tf_gaussian, normalize_patch=False):
        self.nr = nr
        self.l_max = l_max
        self.rad = rad
        self.radial_fn = radial_fn
        # self.radial_fn = tf_zero
        self.normalize_patch = normalize_patch

    def compute(self, X, sq_dist):
        return tf_sh_kernel(X, sq_dist,
                            self.nr,
                            self.l_max,
                            self.rad,
                            self.radial_fn,
                            normalize_patch=self.normalize_patch,
                            dtype=tf.float32)

    def get_shape(self):
        return ((self.l_max + 1)**2, self.nr)


def shconv(x, patches_idx, shkernel):
    x = tf.gather_nd(x, patches_idx)
    y = tf.einsum('bvpmc,bvpnr->bvmnrc', x, shkernel)
    y = tf.reshape(y, (y.shape[0], y.shape[1], y.shape[2], y.shape[3], -1))
    return y





