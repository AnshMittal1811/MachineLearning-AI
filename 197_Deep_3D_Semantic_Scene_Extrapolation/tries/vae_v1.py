import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data

tfd = tf.contrib.distributions

def make_encoder(data, code_size):
  x = tf.layers.flatten(data)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  loc = tf.layers.dense(x, code_size)
  scale = tf.layers.dense(x, code_size, tf.nn.softplus)
  return tfd.MultivariateNormalDiag(loc, scale)


def make_prior(code_size):
  loc = tf.zeros(code_size)
  scale = tf.ones(code_size)
  return tfd.MultivariateNormalDiag(loc, scale)


def make_decoder(code, data_shape):
  x = code
  x = tf.layers.dense(x, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
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


data = tf.placeholder(tf.float32, [None, 28, 28])

make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

# Define the model.
prior = make_prior(code_size=2)
posterior = make_encoder(data, code_size=2)
code = posterior.sample()

# Define the loss.
likelihood = make_decoder(code, [28, 28]).log_prob(data)
divergence = tfd.kl_divergence(posterior, prior)
elbo = tf.reduce_mean(likelihood - divergence)
optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)

samples = make_decoder(prior.sample(10), [28, 28]).mean()

mnist = input_data.read_data_sets('MNIST_data/')
#fig, ax = plt.subplots(nrows=20, ncols=11, figsize=(10, 20))
with tf.train.MonitoredSession() as sess:
  for epoch in range(20):
    feed = {data: mnist.test.images.reshape([-1, 28, 28])} 
    test_elbo, test_codes, test_samples = sess.run([elbo, code, samples], feed)
    print('Epoch', epoch, 'elbo', test_elbo)
    #ax[epoch, 0].set_ylabel('Epoch {}'.format(epoch))
    #plot_codes(ax[epoch, 0], test_codes, mnist.test.labels)
    #plot_samples(ax[epoch, 1:], test_samples)
    for _ in range(600):
      feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])} 
      #####################################
      # temp = mnist.train.next_batch(100)[0]
      # print temp.shape
      # print temp[0][300:400]
      # sys.exit(0)
      #####################################
      
      sess.run(optimize, feed)
#plt.savefig('vae-mnist.png', dpi=300, transparent=True, bbox_inches='tight')