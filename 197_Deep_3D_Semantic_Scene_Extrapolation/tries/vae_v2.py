import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os, random
import glob
from   random           import randint
import utils
# from tensorflow.examples.tutorials.mnist import input_data



classes_count        = 14
scene_shape          = [84, 84] 
halfed_scene_shape   = scene_shape[1] / 2  
directory            = 'vae_v2'
to_train             = True
to_restore           = False
show_accuracy        = True
show_accuracy_step   = 500
save_model           = True
save_model_step      = 3000
visualize_scene      = True
visualize_scene_step = 5000
subset_train         = False 
train_directory      = 'house_2d/' 
test_directory       = 'house_2d/'
max_iter             = 500000
learning_rate        = 0.00005
batch_size           = 64 
num_of_vis_batch     = 1

# logging.basicConfig(filename=str(directory)+'.log', level=logging.DEBUG) 

if not os.path.exists(directory):
    os.makedirs(directory)
    
train_data = []  
test_data  = []

for item in glob.glob(train_directory + "*.npy"):
    train_data.append(item)
    
for item in glob.glob(test_directory + "*.npy"):
    test_data.append(item)

batch_threshold = 0
if subset_train:
    batch_threshold = batch_size * visualize_scene_step
else:
    batch_threshold = len(train_data)
    
    
    
    
    

tfd = tf.contrib.distributions

def make_encoder(data, code_size):
    x = tf.layers.flatten(data)
    x = tf.layers.dense(x, 200, tf.nn.relu)    ###
    x = tf.layers.dense(x, 200, tf.nn.relu)    ###
    loc = tf.layers.dense(x, code_size)
    scale = tf.layers.dense(x, code_size, tf.nn.softplus)
    return tfd.MultivariateNormalDiag(loc, scale)


def make_prior(code_size):
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    return tfd.MultivariateNormalDiag(loc, scale)


def make_decoder(code, data_shape):
    x = code
    x = tf.layers.dense(x, 200, tf.nn.relu)   ###
    x = tf.layers.dense(x, 200, tf.nn.relu)   ###
    logit = tf.layers.dense(x, np.prod(data_shape))
    logit = tf.reshape(logit, [-1] + data_shape)
    return tfd.Independent(tfd.Bernoulli(logit), 2)


def plot_codes(ax, codes, labels):
    ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min() - .1, codes.max() + .1)
    ax.set_ylim(codes.min() - .1, codes.max() + .1)
    ax.tick_params(
            axis='both', which='both', left='off', bottom='off',
            labelleft='off', labelbottom='off')

            
def plot_samples(ax, samples):
    for index, sample in enumerate(samples):
        ax[index].imshow(sample, cmap='gray')
        ax[index].axis('off')
        
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

    x = np.reshape(x, (-1, scene_shape[0], scene_shape[1]))
    y = np.reshape(y, (-1, scene_shape[0], scene_shape[1]))

    return x, y


data = tf.placeholder(tf.float32, [None, 84, 84])

make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

# Define the model.
prior = make_prior(code_size=2)
posterior = make_encoder(data, code_size=2)
code = posterior.sample()

# Define the loss.
likelihood = make_decoder(code, [84, 84]).log_prob(data)
divergence = tfd.kl_divergence(posterior, prior)
elbo = tf.reduce_mean(likelihood - divergence)
optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)

samples = make_decoder(prior.sample(10), [84, 84]).mean() 

# mnist = input_data.read_data_sets('MNIST_data/')
#fig, ax = plt.subplots(nrows=20, ncols=11, figsize=(10, 20))
with tf.train.MonitoredSession() as sess:
    for iter in range(5000000):
        if iter%10000 == 0:
            x_batch, y_batch = fetch_x_y(train_data, batch_threshold)  
            x_batch /= 14.0   
            feed = {data: x_batch} 
            test_elbo, test_codes, test_samples = sess.run([elbo, code, samples], feed) 
            test_samples = (test_samples * 14).astype(int)
            for i in range(test_samples.shape[0]):
                utils.npy_to_ply(str(i) + "_generated", test_samples[i]) 
            print('Step', iter, 'elbo', test_elbo) 
        
        x_batch, y_batch = fetch_x_y(train_data, batch_threshold)  
        x_batch /= 14.0 
        feed = {data: x_batch} 
        sess.run(optimize, feed) 
        
#plt.savefig('vae-mnist.png', dpi=300, transparent=True, bbox_inches='tight')