#The Training scrpit is inspired from Image Captioning with Attension Model by Tensorflow

import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import sys
import time
import json
from glob import glob
from PIL import Image
import pickle
import I2S_Model
import I2S_Data
from datetime import datetime

#Allocating the GPU in use
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpus = tf.config.experimental.list_physical_devices('GPU')

#Setting up memory growth
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

#Invoking stdout as training report.
f = open('Training_Report.txt' , 'w')
sys.stdout = f

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Network Started", flush=True)

#Load the Data
PATH = '/path/to/train_images/'
total_data = 4500000
img_name_train, img_name_val,cap_train,cap_val,tokenizer,max_length,image_features_extract_model = I2S_Data.data_loader(PATH,total_data)
print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Train Images size: ",len(img_name_train),"Train Smiles size", len(cap_train),"Test Images size",len(img_name_val),"Test Smiles size", len(cap_val), flush=True)


#Setting up training parameters, found after optimizing
EPOCHS = 25
BATCH_SIZE =600
BUFFER_SIZE = 1000
embedding_dim = 600
units = 1024
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE

#Here, we are using Inception V3 as base so the feature shape is set to 2048 and the attention shape is set to 64
features_shape = 2048
attention_features_shape = 64

#Loading Numpy files transcibed from PNG images
def map_func(img_name, cap):
	img_tensor = np.load(img_name.decode('utf-8')+'.npy')
	return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)

# shuffling and batching
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

encoder = I2S_Model.CNN_Encoder(embedding_dim)
decoder = I2S_Model.RNN_Decoder(embedding_dim, units, vocab_size)

#Network Parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00051)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_mean(loss_)

#Setting up path to save checkpoint
checkpoint_path = "./checkpoints/train_4_5_mil"
ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)

start_epoch = 0
#Loading checkpoint to last saved
if ckpt_manager.latest_checkpoint:
	ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
	start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

#the loss_plot array will be reset many times
loss_plot = []

#Main Train function wrapped around Tf funtion
@tf.function
def train_step(img_tensor, target):
	loss = 0

	# initializing the hidden state for each batch because the captions are not related from image to image
	hidden = decoder.reset_state(batch_size=target.shape[0])

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

	with tf.GradientTape() as tape:
		features = encoder(img_tensor)

		for i in range(1, target.shape[1]):

			predictions, hidden, _ = decoder(dec_input, features, hidden)

			loss += loss_function(target[:, i], predictions)

			dec_input = tf.expand_dims(target[:, i], 1)

	total_loss = (loss / int(target.shape[1]))

	trainable_variables = encoder.trainable_variables + decoder.trainable_variables

	gradients = tape.gradient(loss, trainable_variables)

	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return loss, total_loss

print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Actual Training Started", flush=True)

for epoch in range(start_epoch, EPOCHS):
	start = time.time()
	total_loss = 0

	for (batch, (img_tensor, target)) in enumerate(dataset):
		batch_loss, t_loss = train_step(img_tensor, target)
		total_loss += t_loss

		if batch % 100 == 0:
			print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])), flush=True)
	# storing the epoch end loss value to plot later
	loss_plot.append(total_loss / num_steps)
	
	if epoch % 1 == 0:
		ckpt_manager.save()

	print ('Epoch {} Loss {:.6f}'.format(epoch + 1,total_loss/num_steps), flush=True)
	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),'Time taken for 1 epoch {} sec\n'.format(time.time() - start), flush=True)

#Saving loss plot
plt.plot(loss_plot , '-o', label= "Loss value")
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.show()
plt.gcf().set_size_inches(20, 20)
plt.savefig("Lossplot.png")
plt.close()

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'), "Network Completed", flush=True)
f.close()
