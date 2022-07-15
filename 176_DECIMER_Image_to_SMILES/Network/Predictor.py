import tensorflow as tf
import os
import sys
import time
import pickle
import I2S_Model
import I2S_evalData
from datetime import datetime
import argparse
import deepsmiles

parser = argparse.ArgumentParser(description="Predicting test images")
# Input Arguments
parser.add_argument(
	'--input',
	help = 'Enter the input filename',
	required = True
	)

args= parser.parse_args()

f = open('Predictions.txt' , 'w')
sys.stdout = f
print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Predictions\n\n", flush=True)

tokenizer = pickle.load(open("tokenizer.pkl","rb"))


#Prediction model parameters
embedding_dim = 600
units = 1024
vocab_size = len(tokenizer.word_index) + 1

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

encoder = I2S_Model.CNN_Encoder(embedding_dim)
decoder = I2S_Model.RNN_Decoder(embedding_dim, units, vocab_size)

#Initialize DeepSMILES
converter = deepsmiles.Converter(rings=True, branches=True)


#Evaluator
def evaluate(image):
	maxlength = 74 #should be determined from running the training script

	hidden = decoder.reset_state(batch_size=1)

	temp_input = tf.expand_dims(I2S_evalData.load_image(image)[0], 0)
	img_tensor_val = I2S_evalData.image_features_extract_model(temp_input)
	img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

	features = encoder(img_tensor_val)

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
	result = []

	for i in range(maxlength):
		predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

		predicted_id = tf.argmax(predictions[0]).numpy()
		result.append(tokenizer.index_word[predicted_id])

		if tokenizer.index_word[predicted_id] == '<end>':
			return result

		dec_input = tf.expand_dims([predicted_id], 0)

	return result


#Predictor
def predictor(Image_path):
	
	checkpoint_path = "../Trained_Models"
	ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

	if ckpt_manager.latest_checkpoint:
		ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
		start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

	result = evaluate(Image_path)

	return result

result = predictor(args.input)
print (converter.decode(''.join(result).replace("<start>","").replace("<end>","")),'\tPredictedSmiles', flush=True)

f.close()