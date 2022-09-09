import tensorflow as tf

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, transpose_b=False):

    shape = input_.get_shape().as_list()
    if not transpose_b:
        w_shape = [shape[1], output_size]
    else:
        w_shape = [output_size, shape[1]]

    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable('w', w_shape, tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('b', [output_size],
            initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix, transpose_b=transpose_b) + bias

def conv2d(input_, out_channels, data_format, kernel=5, stride=2, stddev=0.02, name="conv2d"):

    if data_format == "NHWC":
        in_channels = input_.get_shape()[-1]
        strides = [1, stride, stride, 1]
    else: # NCHW
        in_channels = input_.get_shape()[1]
        strides = [1, 1, stride, stride]

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME', data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases, data_format=data_format), conv.get_shape())

        return conv

def conv2d_transpose(input_, output_shape, data_format, kernel=5, stride=2, stddev=0.02,
                     name="conv2d_transpose"):

    if data_format == "NHWC":
        in_channels = input_.get_shape()[-1]
        out_channels = output_shape[-1]
        strides = [1, stride, stride, 1]
    else:
        in_channels = input_.get_shape()[1]
        out_channels = output_shape[1]
        strides = [1, 1, stride, stride]

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, out_channels, in_channels],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=strides, data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases, data_format=data_format), deconv.get_shape())

        return deconv
       

def lrelu(x, alpha=0.2, name="lrelu"):
    with tf.name_scope(name):
      return tf.maximum(x, alpha*x)
