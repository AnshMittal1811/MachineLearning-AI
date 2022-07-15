import tensorflow as tf
import numpy as np

class EncoderConv1D(tf.keras.Model):

    def __init__(self, latent_dim, num_samples=16, name='encoder_conv1d'):
        super(EncoderConv1D, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.num_samples = num_samples

        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=3)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=3)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')

        self.conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=7, strides=3)
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.Activation('relu')

        self.dense1 = tf.keras.layers.Dense(512)
        self.dense_norm1 = tf.keras.layers.BatchNormalization()
        self.dense_act1 = tf.keras.layers.Activation('relu')

        self.z_mean_dense = tf.keras.layers.Dense(self.latent_dim)
        self.z_log_var_dense = tf.keras.layers.Dense(self.latent_dim)

    def call(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.conv1(inputs)
            x = self.norm1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = self.norm2(x)
            x = self.act2(x)

            x = self.conv3(x)
            x = self.norm3(x)
            x = self.act3(x)
            x = tf.keras.layers.Flatten()(x)

            z_mean = self.z_mean_dense(x)
            z_log_var = self.z_log_var_dense(x)

            self.dist = tf.distributions.Normal(loc=z_mean, scale=tf.exp(0.5*z_log_var))
            sampled = self.dist.sample(self.num_samples)
            z = tf.transpose(sampled, [1, 0, 2])
            return z, z_mean, z_log_var

class DecoderCuDNNGRU(tf.keras.Model):

    def __init__(self, charset_length, max_length, name='decoder_gru'):
        super(DecoderCuDNNGRU, self).__init__(name=name)
        self.charset_length = charset_length
        self.max_length = max_length

        self.dense1 = tf.keras.layers.Dense(512)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.repeat = tf.keras.layers.RepeatVector(self.max_length)

        self.gru1 = tf.keras.layers.CuDNNGRU(512, return_sequences=True)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.gru2 = tf.keras.layers.CuDNNGRU(512, return_sequences=True)
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.gru3 = tf.keras.layers.CuDNNGRU(512, return_sequences=True)
        self.out_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.charset_length))
        self.out_act = tf.keras.layers.Activation('softmax')


    def call(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.dense1(inputs)
            x = self.norm1(x)
            x = self.act1(x)

            x = self.repeat(x)
            x = self.gru1(x)
            x = self.norm2(x)
            x = self.gru2(x)
            x = self.norm3(x)
            x = self.gru3(x)
            outputs_logits = self.out_dense(x)
            outputs = self.out_act(outputs_logits)
            return outputs, outputs_logits

class VariationalAutoencoder(tf.keras.Model):

    def __init__(self, latent_dim, charset_length, max_length, num_samples=16, name='vae'):
        super(VariationalAutoencoder, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.charset_length = charset_length
        self.max_length = max_length
        self.num_samples = num_samples

        self.encoder = EncoderConv1D(self.latent_dim, num_samples=self.num_samples)
        self.decoder = DecoderCuDNNGRU(charset_length, max_length)

    def call(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            z, self.z_mean, self.z_log_var = self.encoder(inputs)
            z_reshaped = tf.reshape(z, (-1, self.encoder.latent_dim))
            outputs, self.outputs_logits = self.decoder(z_reshaped)
            return outputs

    def vae_loss_func(self, y_true, y_pred):
        latent_loss = -0.5*tf.reduce_sum(1.0 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), 1)
        y_true_r = tf.reshape(y_true, [-1, 1, self.max_length])
        y_true_c = tf.cast(y_true_r, tf.int64)
        tiled = tf.tile(y_true_c, (1, self.num_samples, 1))
        y_true_rep = tf.reshape(tiled, (-1, self.max_length))
        recon_loss = tf.losses.sparse_softmax_cross_entropy(y_true_rep, self.outputs_logits, reduction=tf.losses.Reduction.SUM)
        recon_loss = recon_loss/tf.cast(self.num_samples, tf.float32)
        vae_loss = latent_loss + recon_loss
        return vae_loss

    def sampled_data_acc(self, y_true, y_pred):
        y_true_r = tf.reshape(y_true, [-1, 1, self.max_length])
        y_true_c = tf.cast(y_true_r, tf.int64)
        tiled = tf.tile(y_true_c, (1, self.num_samples, 1))
        y_true_rep = tf.reshape(tiled, (-1, self.max_length))
        y_pred_class = tf.argmax(y_pred, axis=-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_true_rep, y_pred_class), tf.float32))
        return acc
