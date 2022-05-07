import tensorflow as tf
from tensorflow.python.keras import backend as K


class Jitter(tf.keras.layers.Layer):
    def __init__(self, scale, training=None):
        super(Jitter, self).__init__()
        self.scale = scale
        self.training = training

    def build(self, input_shape):
        pass

    def call(self, x, training=None):
        if training is None:
            training = K.learning_phase()
        if training:
            g = tf.random.normal(shape=x.shape, stddev=1./3.)
            gn = tf.norm(g, axis=-1, keepdims=True)
            g = self.scale*tf.divide(g, tf.maximum(gn, 1.0))
            return tf.add(g, x)
        else:
            return x

