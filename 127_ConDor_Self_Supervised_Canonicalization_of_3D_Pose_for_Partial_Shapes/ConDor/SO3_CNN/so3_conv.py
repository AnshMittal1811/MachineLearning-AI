import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D


class SO3Conv(tf.keras.layers.Layer):
    def __init__(self,
               units,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               name_='',
               **kwargs):
        super(SO3Conv, self).__init__(**kwargs)
        self.name_ = name_
        self.units = int(units) if not isinstance(units, int) else units
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        self.kernels = dict()
        for l in input_shape:
            if l.isnumeric():
                self.kernels[l] = self.add_weight(
                                        self.name_ + '_kernel_'+l,
                                        shape=[2*int(l)+1, 2*int(l)+1, input_shape[l][-1], self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                self.name_ + '_bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, x):
        y = dict()
        for l in x:
            if l.isnumeric() and l in self.kernels:
                y[l] = tf.einsum('...ikc,jkcd->...ijd', x[l], self.kernels[l])


        if self.bias is not None and '0' in y:
            y['0'] = tf.nn.bias_add(y['0'], self.bias)

        return y
