from Reader_data import Reader
from bert_vectorizer import BERTVectorizer
from joint_bert_model import JointBertModel
from itertools import chain

import argparse
import os
import pickle
import tensorflow as tf
from sklearn import metrics
import json

parser = argparse.ArgumentParser('Evaluating the Joint BERT model')
parser.add_argument('--model', '-m', help='path to joint bert model', type=str, required=True)
parser.add_argument('--data', '-d', help='path to test data', type=str, required=True)
parser.add_argument('--batch', '-bs', help='batch size', type=int, default=128, required=False)
parser.add_argument('--pre_intents', '-pre_is', help='teh file name of saving predicted intents', type=str, required=True)
parser.add_argument('--pre_slots', '-pre_sls', help='the file name of saving predicted slots/tags', type=str, required=True)

args = parser.parse_args()
load_folder_path = args.model
data_folder_path = args.data
batch_size = args.batch
pre_intens_name = args.pre_intents
pre_slots_name = args.pre_slots

sess = tf.compat.v1.Session()

bert_model_hub_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)

## loading the model
print('Loading models ....')
if not os.path.exists(load_folder_path):
    print('Folder "%s" not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
    tags_vectorizer = pickle.load(handle)
    slots_num = len(tags_vectorizer.label_encoder.classes_)
with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
    intents_label_encoder = pickle.load(handle)
    intents_num = len(intents_label_encoder.classes_)

model = JointBertModel.load(load_folder_path, sess)

data_text_arr, data_tags_arr, data_intents = Reader.read(data_folder_path)
data_input_ids, data_input_mask, data_segment_ids, data_valid_positions, data_sequence_lengths = bert_vectorizer.transform(data_text_arr)


def flatten(y):
    ## flatten a list of lists.
    ## flatten([[1,2], [3,4]]) --> [1, 2, 3, 4]
    return list(chain.from_iterable(y))

def get_results(input_ids, input_mask, segment_ids, valid_positions, sequence_lengths, tags_arr, intents, tags_vectorizer, intents_label_encoder):
    predicted_tags, predicted_intents = model.predict_slots_intent(
        [input_ids, input_mask, segment_ids, valid_positions],
        tags_vectorizer, intents_label_encoder, remove_start_end=True
    )
    real_tags = [x.split() for x in tags_arr]

    f1_score = metrics.f1_score(flatten(real_tags), flatten(predicted_tags), average='micro')
    acc = metrics.accuracy_score(intents, predicted_intents)
    return f1_score, acc, predicted_intents, predicted_tags


print('.....Evaluation....')
f1_score, acc, predicted_intents, predicted_tags = get_results(data_input_ids, data_input_mask, data_segment_ids, data_valid_positions, data_sequence_lengths,
                            data_tags_arr, data_intents, tags_vectorizer, intents_label_encoder)


### print(type(predicted_tags)) ## <class 'numpy.ndarray'>
### save the predicted slots to file
with open(os.path.join(load_folder_path, pre_slots_name), 'w') as fp:
    for item in predicted_tags:
        #print(item)
        fp.write(" ".join(item) + "\n")
    fp.close()

### print(type(predicted_intents)) ## <class 'numpy.ndarray'>
### save the predicted intents to file
with open(os.path.join(load_folder_path, pre_intens_name), 'w') as fp:
    for item in predicted_intents:
        #print(item)
        fp.write("".join(map(str, item)))
    fp.close()

print('Slot f1 score = %f' % f1_score)
print('Intent accuracy = %f' % acc)


### save the f1 score and accuracy to file
eva_results = {
    "slots_f1_score" : f1_score,
    "intent_accuracy" : acc
}
with open(os.path.join(load_folder_path, "eva_results.json"), 'w') as json_file:
    json.dump(eva_results, json_file)

tf.compat.v1.reset_default_graph()

