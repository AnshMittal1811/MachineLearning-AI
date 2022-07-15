import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import re
import os
import sys
import time
import json
from glob import glob
from PIL import Image
import pickle
import I2S_Model
import I2S_evalData
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

f = open('Test_predictions.txt' , 'w')
sys.stdout = f
#print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Network Started", flush=True)

#Load the Data
PATH = 'path/to/Test_Images/'

img_name_val,cap_val,tokenizer,max_length,image_features_extract_model = I2S_evalData.data_loader(PATH)

#Necessary Parameters
embedding_dim = 600
units = 1024
vocab_size = len(tokenizer.word_index) + 1

features_shape = 2048
attention_features_shape = 64


#Loading network model
encoder = I2S_Model.CNN_Encoder(embedding_dim)
decoder = I2S_Model.RNN_Decoder(embedding_dim, units, vocab_size)

#Loading checkpoints (Trained Model)
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
	ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
	start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

#Evaluator
def evaluate(image):
	hidden = decoder.reset_state(batch_size=1)

	temp_input = tf.expand_dims(I2S_evalData.load_image(image)[0], 0)
	img_tensor_val = image_features_extract_model(temp_input)
	img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

	features = encoder(img_tensor_val)

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
	result = []

	for i in range(max_length):
		predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

		predicted_id = tf.argmax(predictions[0]).numpy()
		result.append(tokenizer.index_word[predicted_id])

		if tokenizer.index_word[predicted_id] == '<end>':
			return result

		dec_input = tf.expand_dims([predicted_id], 0)

	return result

# Predicting Smiles on the validation set
for k in range(len(img_name_val)):
	image = img_name_val[k]
	real_caption = ''.join(cap_val[k])
	result = evaluate(image)

	print (real_caption.replace(" ","").replace("<start>","").replace("<end>",""),'\tOriginalSmiles', flush=True)
	print (''.join(result).replace("<start>","").replace("<end>",""),'\tPredictedSmiles', flush=True)
#print("Predictions Completed!")
f.close()