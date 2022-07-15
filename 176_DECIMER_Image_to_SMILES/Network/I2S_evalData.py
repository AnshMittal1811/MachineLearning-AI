import tensorflow as tf
import time
import json
from glob import glob
from PIL import Image
import pickle
import I2S_Data

#Create Inception V3 Model
image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)	

def load_image(image_path):
	img = tf.io.read_file(image_path)
	img = tf.image.decode_png(img, channels=3)
	img = tf.image.resize(img, (299, 299))
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	return img, image_path

def data_loader(PATH):
	
	# read the Captions file
	with open('Test_captions.txt', 'r') as txt_file:
		smiles = txt_file.read()

	# storing the captions and the image name in vectors
	all_smiles = []
	all_img_name = []

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

	test_captions, img_name_vector = (all_smiles,all_img_name)

	print("Selected Test Data: ",len(test_captions),"Total data in use: ", len(all_smiles), flush=True)


	#Create Inception V3 Model
	image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
	new_input = image_model.input
	hidden_layer = image_model.layers[-1].output
	image_features_extract_model = tf.keras.Model(new_input, hidden_layer)		

	#Defining the maximum number of tokens to generate
	def calc_max_length(tensor):
		return max(len(t) for t in tensor)

	PATH = os.path.abspath('.')+'/Train_Images/'
	total_data = 16780000
	img_name_train, img_name_val,cap_train,cap_val,tokenizer,max_length,image_features_extract_model = I2S_Data.data_loader(PATH,total_data)

	tokenizer.word_index['<pad>'] = 0
	tokenizer.index_word[0] = '<pad>'

	return img_name_vector,test_captions,tokenizer,max_length,image_features_extract_model
