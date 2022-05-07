import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

"""
def batch_vm(v, m):
  shape = tf.shape(v)
  rank = shape.get_shape()[0].value
  v = tf.expand_dims(v, rank)

  vm = tf.mul(v, m)

  return tf.reduce_sum(vm, rank-1)

def batch_vm2(x, m):
  [input_size, output_size] = m.get_shape().as_list()

  input_shape = tf.shape(x)
  batch_rank = input_shape.get_shape()[0].value - 1
  batch_shape = input_shape[:batch_rank]
  output_shape = tf.concat(0, [batch_shape, [output_size]])

  x = tf.reshape(x, [-1, input_size])
  y = tf.matmul(x, m)

  y = tf.reshape(y, output_shape)

  return y
"""

def align_patches(f_patches, transport):
    transport = tf.expand_dims(transport, axis=3)
    f_patches = tf.expand_dims(f_patches, axis=-2)
    return tf.reduce_sum(tf.multiply(transport, f_patches), axis=-1, keepdims=False)

def md_polar(x):
    x_norm = tf.reduce_sum(tf.multiply(x, x), axis=-1, keepdims=True)
    x_norm = tf.sqrt(tf.maximum(x_norm, 0.000001))
    x_dir = tf.divide(x, x_norm)
    return x_norm, x_dir

"""
def tf_frames(n):
    shape = n.get_shape().as_list()
    rank = len(shape)
    zero = tf.zeros(shape[:-2])
    u0 = tf.stack([zero, n[..., 2], -n[..., 1]], axis=-1)
    u1 = tf.stack([-n[..., 2], zero, n[..., 0]], axis=-1)
    u2 = tf.stack([n[..., 1], -n[..., 0], zero], axis=-1)
    Q = tf.stack([u0, u1, u2], axis=-2)
    Q2 = tf.matmul(Q, Q)
    c = n[..., 2]
    # s = tf.sqrt(tf.maximum(n[..., 0]*n[..., 0] + n[..., 1]*n[..., 1], 0.00001))
    c_1 = tf.reciprocal(tf.maximum(1.+c, 0.0001))
    c_1 = tf.reshape(c_1 , shape[:-2] + [1, 1])
    I = tf.reshape(tf.eye(3), [1]*(rank-1) + [3, 3])

    return tf.add(I, tf.add(Q, tf.multiply(c_1, Q2)))
"""

def tf_frames(n):
    shape = n.get_shape().as_list()
    zero = tf.zeros(shape[:-1])
    u0 = tf.stack([zero, n[..., 2], -n[..., 1]], axis=-1)
    u1 = tf.stack([-n[..., 2], zero, n[..., 0]], axis=-1)
    u2 = tf.stack([n[..., 1], -n[..., 0], zero], axis=-1)

    Q = tf.stack([u0, u1, u2], axis=-2)
    Q_norm2 = tf.multiply(Q, Q)
    Q_norm2 = tf.reduce_sum(Q_norm2, axis=-1, keepdims=False)
    idx = tf.argmax(Q_norm2, axis=-1, output_type=tf.int32)

    batch_size = shape[0]
    num_points = shape[1]
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1,))
    batch_idx = tf.tile(batch_idx, (1, num_points,))

    point_idx = tf.range(0, num_points)
    point_idx = tf.reshape(point_idx, (1, num_points))
    point_idx = tf.tile(point_idx, (batch_size, 1))

    indices = tf.stack([batch_idx, point_idx, idx], -1)
    u = tf.gather_nd(Q, indices)
    # normalize u
    u = tf.linalg.l2_normalize(u, axis=-1)
    v = tf.cross(n, u)
    return tf.stack([u, v, n], axis=-1)


def tf_transport_angles_(frames, frame_patches):
    batch_size = frame_patches.get_shape()[0].value
    num_points = frame_patches.get_shape()[1].value
    patch_size = frame_patches.get_shape()[2].value
    # frames = frame_patches[:, :, 0, ...]
    # frames = tf.expand_dims(frames, axis=2)
    n = frames[..., 2]
    n = tf.expand_dims(n, axis=2)
    n = tf.tile(n, (1, 1, patch_size, 1))
    n_patches = frame_patches[..., 2]
    q = tf.linalg.cross(n, n_patches)
    # c = tf.tensordot(n, n_patches, [[-1], [-1]])
    c = tf.reduce_sum(tf.multiply(n, n_patches), axis=-1, keepdims=False)

    mask = tf.maximum(tf.sign(c + 0.9), 0.)

    c_1 = tf.reciprocal(tf.maximum(1. + c, 0.0001))
    c_1 = tf.reshape(c_1, (batch_size, num_points, patch_size, 1, 1))

    zero = tf.zeros(c.get_shape())
    u0 = tf.stack([zero, q[..., 2], -q[..., 1]], axis=-1)
    u1 = tf.stack([-q[..., 2], zero, q[..., 0]], axis=-1)
    u2 = tf.stack([q[..., 1], -q[..., 0], zero], axis=-1)
    Q = tf.stack([u0, u1, u2], axis=-2)
    Q2 = tf.matmul(Q, Q)
    I = tf.reshape(tf.eye(3), (1, 1, 1, 3, 3))
    R = tf.add(I, tf.add(Q, tf.multiply(c_1, Q2)))
    transported_frames = tf.einsum('bvpij,bvjk->bvpik', R, frames[..., :-1])
    # R = tf.matmul(frame_patches[..., :-1], tf.expand_dims(R[..., 0], axis=-1), transpose_a=True)
    R = tf.matmul(frame_patches[..., :-1], transported_frames, transpose_a=True)
    return tf.atan2(R[..., 0, 0], R[..., 1, 0]), mask


def tf_transport_angles(frames, indices):
    batch_size = frames.get_shape()[0].value
    num_points = frames.get_shape()[1].value
    patch_size = indices.get_shape()[-1].value
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, num_points, patch_size))
    indices = tf.stack([batch_idx, indices], -1)
    frame_patches = tf.gather_nd(frames, indices)
    return tf_transport_angles_(frames, frame_patches)


def tf_transport_(frames, frame_patches, k):
    theta, mask = tf_transport_angles_(frames, frame_patches)
    mask = tf.expand_dims(mask, axis=-1)
    theta = - theta
    theta = tf.expand_dims(theta, axis=-1)
    theta = tf.tile(theta, (1, 1, 1, k))
    K = tf.range(k, dtype=tf.float32) + 1.
    K = tf.reshape(K, (1, 1, 1, k))
    theta = tf.multiply(K, theta)
    c = tf.cos(theta)
    c = tf.multiply(mask, c)
    s = tf.sin(theta)
    s = tf.multiply(mask, s)
    R0 = tf.stack([c, -s], axis=-1)
    R1 = tf.stack([s, c], axis=-1)
    return tf.stack([R0, R1], axis=-2)


def tf_transport(frames, indices, k):
    """
    batch_size = frames.get_shape()[0].value
    num_points = frames.get_shape()[1].value
    patch_size = indices.get_shape()[-1].value
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, num_points, patch_size))
    indices = tf.stack([batch_idx, indices], -1)
    """
    frame_patches = tf.gather_nd(frames, indices)
    return tf_transport_(frames, frame_patches, k)

"""
def tf_transport_angles(frames, indices):
    batch_size = frames.get_shape()[0].value
    num_points = frames.get_shape()[1].value
    patch_size = indices.get_shape()[-1].value
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, num_points, patch_size))
    indices = tf.stack([batch_idx, indices], -1)
    frame_patches = tf.gather_nd(frames, indices)
    frames = tf.expand_dims(frames, axis=2)
    n = frames[..., 2]
    n = tf.tile(n, (1, 1, patch_size))
    n_patches = frame_patches[..., 2]
    q = tf.linalg.cross(n, n_patches)
    c = tf.tensordot(n, n_patches, [[-1], [-1]])

    mask = tf.maximum(tf.sign(c + 0.9), 0.)

    c_1 = tf.reciprocal(tf.maximum(1. + c, 0.0001))
    c_1 = tf.reshape(c_1, (batch_size, num_points, patch_size, 1, 1))

    zero = tf.zeros(c.get_shape())
    u0 = tf.stack([zero, q[..., 2], -q[..., 1]], axis=-1)
    u1 = tf.stack([-q[..., 2], zero, q[..., 0]], axis=-1)
    u2 = tf.stack([q[..., 1], -q[..., 0], zero], axis=-1)
    Q = tf.stack([u0, u1, u2], axis=-2)
    Q2 = tf.matmul(Q, Q)
    I = tf.reshape(tf.eye(3), (1, 1, 1, 3, 3))
    R = tf.add(I, tf.add(Q, tf.multiply(c_1, Q2)))
    R = tf.matmul(frame_patches[..., :-2], tf.expand_dims(R[..., 0], axis=-1), transpose_a=True)
    return tf.atan2(R[..., 0, 0], R[..., 1, 0]), mask


def tf_transport(frames, indices, k):
    theta, mask = tf_transport_angles(frames, indices)
    mask = tf.expand_dims(mask, axis=-1)
    theta = - theta
    theta = tf.expand_dims(theta, axis=-1)
    theta = tf.tile(theta, (1, 1, 1, k+1))
    k = tf.range(k+1, dtype=tf.float32)
    k = tf.reshape(k, (1, 1, 1, k+1))
    theta = tf.multiply(k, theta)
    c = tf.cos(theta)
    c = tf.multiply(mask, c)
    s = tf.sin(theta)
    s = tf.multiply(mask, s)
    R0 = tf.stack([c, -s], axis=-1)
    R1 = tf.stack([s, c], axis=-1)
    return tf.stack([R0, R1], axis=-2)
"""

class Frames(Layer):
    def __init__(self, **kwargs):
        super(Frames, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Frames, self).build(input_shape)

    def call(self, x):
        frames = tf_frames(x)
        return frames

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 3)


class Transport(Layer):
    def __init__(self, num_fourier, **kwargs):
        self.num_fourier = num_fourier
        super(Transport, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(Transport, self).build(input_shape)

    def call(self, x):
        transport = tf_transport(x[0], x[1], self.num_fourier)
        return transport

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[1][0], input_shape[1][1], input_shape[1][2], self.num_fourier, 2, 2)


class FramesAndTransport(Layer):

    def __init__(self, num_fourier, **kwargs):
        self.num_fourier = num_fourier
        super(FramesAndTransport, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(FramesAndTransport, self).build(input_shape)

    def call(self, x):
        frames = tf_frames(x[0])
        transport = tf_transport(frames, x[1], self.num_fourier)
        return [frames, transport]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        return [(input_shape[0][0], input_shape[0][1], 3),
                (input_shape[1][0], input_shape[1][1], input_shape[1][2], self.num_fourier, 2, 2)]


class MdActivation(Layer):

    def __init__(self, use_bias, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]