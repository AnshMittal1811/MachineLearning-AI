import tensorflow as tf
import re
import numpy as np
import os
import sys
import time
import json
from glob import glob
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_image(image_path):
	img = tf.io.read_file(image_path)
	img = tf.image.decode_png(img, channels=3)
	img = tf.image.resize(img, (299, 299))
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	return img, image_path

def data_loader(PATH,total_data):

	with open('DeepSMILES_as_output.txt', 'r') as txt_file:
		smiles = txt_file.read()

	all_smiles = []
	all_img_name = []
	
	#Splitting the text file and saving SMILES as Captions
	for line in smiles.split('\n'):
		tokens = line.split(',')
		image_id = str(tokens[0])+'.png'
		try:
			caption = '<start> ' + str(tokens[1].rstrip()) + ' <end>'
		except IndexError as e:
			print (e,flush=True)
		full_image_path = PATH + image_id

		all_img_name.append(full_image_path)
		all_smiles.append(caption)

	train_captions, img_name_vector = (all_smiles,all_img_name)

	selected_data = total_data
	train_captions = train_captions[:selected_data]
	img_name_vector = img_name_vector[:selected_data]
	print("Selected Data ",len(train_captions),"All data ", len(all_smiles), flush=True)

	#Loading InceptionsV3 Model to convert Images to Numpy arrays (features)
	image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')

	new_input = image_model.input
	hidden_layer = image_model.layers[-1].output

	image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

	encode_train = sorted(set(img_name_vector))

	image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
	image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(100)

	for img, path in image_dataset:
		batch_features = image_features_extract_model(img)
		batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))

		for bf, p in zip(batch_features, path):
			try:
				path_of_feature = p.numpy().decode("utf-8")
				np.save(path_of_feature, bf.numpy())
			except OSError as e:
				print (e)
			

	def calc_max_length(tensor):
		return max(len(t) for t in tensor)

	#Maximum number of allowed Vocabulary, Unimg UniqueSMILES.py we can define this a bit early
	max_voc = 100

	#Creating tokenizer with defined characters
	tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_voc,oov_token="<unk>",filters='!"$&:;?^`{}~ ',lower=False)
	tokenizer.fit_on_texts(train_captions)
	train_seqs = tokenizer.texts_to_sequences(train_captions)

	tokenizer.word_index['<pad>'] = 0
	tokenizer.index_word[0] = '<pad>'

	cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
	max_length = calc_max_length(train_seqs)
	
	#Splitting the dataset to train and test, in our case the test set is 10% of the train set. And completely unseen by the training process.
	img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,cap_vector,test_size=0.1,random_state=0,shuffle=False)

	return img_name_train, img_name_val,cap_train,cap_val,tokenizer,max_length,image_features_extract_model
