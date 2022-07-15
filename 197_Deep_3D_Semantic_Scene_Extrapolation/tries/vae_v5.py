import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.slim import conv2d as conv
from tensorflow.contrib.slim import conv2d_transpose as deconv
from tensorflow.contrib.slim import batch_norm as batch_norm
from tensorflow.contrib import slim 

import matplotlib
import sys, os, glob, random
import utils
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# plt.ioff()

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# num_sample = mnist.train.num_examples
input_dim = 84 * 84 
w = h = 84

classes_count        = 14
scene_shape          = [84, 84] 
halfed_scene_shape   = scene_shape[1] / 2  
directory            = 'vae_v5'
to_train             = True
to_restore           = False
show_accuracy        = True
show_accuracy_step   = 500
save_model           = True
save_model_step      = 3000
visualize_scene      = True
visualize_scene_step = 10
subset_train         = False 
train_directory      = 'house_2d/' 
test_directory       = 'house_2d/'
max_iter             = 500000
learning_rate        = 0.00005
batch_size           = 128 
num_of_vis_batch     = 1
if not os.path.exists(directory):
    os.makedirs(directory)
    
train_data = []  
test_data  = []

for item in glob.glob(train_directory + "*.npy"):
    train_data.append(item) 
    # train_data.append(np.load(item)) 

# train_data = np.reshape(train_data, (-1, scene_shape[0] * scene_shape[1])) 
# x_mean = np.mean(train_data, axis = 0)
# x_std = np.std(train_data, axis = 0)

# print train_data.shape
# print x_mean.shape
# print x_std.shape

# np.save("x_mean_2d", x_mean)
# np.save("x_std_2d", x_std)
# sys.exit()
    
for item in glob.glob(test_directory + "*.npy"):
    test_data.append(item)

batch_threshold = 0
if subset_train:
    batch_threshold = batch_size * visualize_scene_step
else:
    batch_threshold = len(train_data) 


class VariantionalAutoencoder(object):

    def __init__(self, learning_rate=1e-3, batch_size=64, n_z=1000):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z

        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f0 = fc(self.x, 4096, scope='enc_fc0', activation_fn=tf.nn.elu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),weights_regularizer=slim.l2_regularizer(0.05))
        f1 = fc(f0,     2048, scope='enc_fc1', activation_fn=tf.nn.elu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),weights_regularizer=slim.l2_regularizer(0.05))
        f2 = fc(f1,     1024, scope='enc_fc2', activation_fn=tf.nn.elu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),weights_regularizer=slim.l2_regularizer(0.05))
        f3 = fc(f2,     512,  scope='enc_fc3', activation_fn=tf.nn.elu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),weights_regularizer=slim.l2_regularizer(0.05))
        
        self.z_mu           = fc(f3, self.n_z, scope='enc_fc4_mu',    activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g0 =         fc(self.z, 512 ,      scope='dec_fc0', activation_fn=tf.nn.elu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),weights_regularizer=slim.l2_regularizer(0.05))
        g1 =         fc(g0,     1048,      scope='dec_fc1', activation_fn=tf.nn.elu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),weights_regularizer=slim.l2_regularizer(0.05))
        g2 =         fc(g1,     2048,      scope='dec_fc2', activation_fn=tf.nn.elu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),weights_regularizer=slim.l2_regularizer(0.05))
        g3 =         fc(g2,     4096,      scope='dec_fc3', activation_fn=tf.nn.elu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),weights_regularizer=slim.l2_regularizer(0.05))
        self.x_hat = fc(g3,     input_dim, scope='dec_fc4', activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat), axis=1)
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x}
        )
        return loss, recon_loss, latent_loss

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z

def fetch_x_y(data, limit):
    batch, x, y = [], [], []  
    random_batch = []
    for i in xrange(batch_size): # randomly fetch batch
        random_batch.append(data[random.randint(0, limit-1)])

    for npyFile in random_batch:   
        batch.append(np.load(npyFile))
        
    batch = np.reshape(batch, (-1, scene_shape[0], scene_shape[1]))    

    x = batch[ :, 0:scene_shape[0], 0:scene_shape[1] ]  # input 
    y = batch[ :, 0:scene_shape[0], 0:scene_shape[1] ]  # gt  

    x = np.reshape(x, (-1, scene_shape[0] * scene_shape[1]))
    y = np.reshape(y, (-1, scene_shape[0],  scene_shape[1]))

    return x, y
    
    
def trainer(learning_rate=1e-5, batch_size=128, num_epoch=75, n_z=1000):
    model = VariantionalAutoencoder(learning_rate=learning_rate, batch_size=batch_size, n_z=n_z)
    
    x_mean = np.load("x_mean_2d.npy")
    x_std = np.load("x_std_2d.npy")

    for iter in range(0, 5000000):
        # Obtina a batch
        x_batch, y_batch = fetch_x_y(train_data, batch_threshold)  
        
        y_batch = np.zeros((batch_size, 84 * 84))
        y_batch[:, :] = x_batch
        y_batch = np.reshape(y_batch, (-1, scene_shape[0], scene_shape[1]))
        
        # normalize X = (X - mean) / std
        x_batch = ((x_batch-x_mean)/x_std*1.0)  
        x_batch_min = np.min(x_batch, axis=1)
        for i in range(batch_size):
            x_batch[i, :] -= x_batch_min[i]
            
        x_batch_max = np.max(x_batch, axis=1)
        for i in range(batch_size):
            x_batch[i, :] /= x_batch_max[i] 
            
        loss, recon_loss, latent_loss = model.run_single_step(x_batch)

        if iter % 10 == 0:
            print('[Iter {}] Loss: {}, Recon: {}, Latent: {}'.format(iter, loss, recon_loss, latent_loss))
            
        if iter % 5000 == 0:    # completion
            x_batch = np.reshape(x_batch, (-1, scene_shape[0], scene_shape[1]))
            x_batch[:, :, :42] = np.random.rand(128, 84, 42) 
            x_batch = np.reshape(x_batch, (-1, scene_shape[0] * scene_shape[1]))
            
            x_reconstructed = model.reconstructor(x_batch) 
            x_reconstructed = np.reshape(x_reconstructed, (-1, scene_shape[0], scene_shape[1])) 
            
            #---------------------------------------- 
            for i in range(batch_size):
                x_reconstructed[i, :] *= x_batch_max[i] 
                
            for i in range(batch_size):
                x_reconstructed[i, :] += x_batch_min[i]
            
            x_reconstructed = np.reshape(x_reconstructed, (-1, scene_shape[0] * scene_shape[1]))            
            x_reconstructed = ((x_reconstructed+x_mean)*x_std*1.0)
            x_reconstructed = np.reshape(x_reconstructed, (-1, scene_shape[0], scene_shape[1]))
            #----------------------------------------
            
            for i in range(20):   
                scene = x_reconstructed[i]
                empty = np.zeros((84, 10))
                scene = np.concatenate((scene, empty), axis=1)
                scene = np.concatenate((scene, y_batch[i]), axis=1)            
                utils.npy_to_ply(directory + "/_" + str(i) + "_generated", scene)  
                
    print('Done!')
    return model

# Train the model
model = trainer(learning_rate=1e-4, batch_size=128, num_epoch=100, n_z=512)
 