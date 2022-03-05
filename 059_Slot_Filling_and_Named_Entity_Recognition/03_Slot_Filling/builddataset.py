import numpy as np
from collections import Counter

def load_data(data_path):
    data = []
    sentence = []
    labels = []
    num_words = 0 
    for line in open(data_path, 'r'):
        line = line.strip()
        if line == '':
            if len(sentence) > 0:
                data.append((sentence, labels))
                sentence = []
                labels = []
        else:
            word, label = line.split()
            num_words += 1
            sentence.append(word)
            labels.append(label)
    if len(sentence) > 0:
        data.append((sentence, labels))
    print("total words num:", num_words)
    print("total sentences num:", len(data))
    return data

def build_label_vocab(slot_names_path):
    labels = ['O']
    for line in open(slot_names_path, 'r'):
        line = line.strip()
        labels.append("B-"+line)
        labels.append("I-"+line)

    label2idx = dict(zip(labels, range(len(labels))))
    idx2label = dict(zip(range(len(labels)), labels))
    return label2idx, idx2label

def build_vocab(data, min_count=1):
    count = [("<UNK>", -1), ("<PAD>", -1)]
    words = []
    for sentence, _ in data: 
        words.extend(sentence)
  
    counter = Counter(words)
    counter_list = counter.most_common()
    for word, c in counter_list:
        if c >= min_count:
            count.append((word, c))
    word2idx = dict()
    for word, _ in count:
        word2idx[word] = len(word2idx)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
  
    return word2idx, idx2word

def build_dataset(data, word2idx, label2idx):
    num_text = []
    num_label = []
    for sentence, label in data:
        num_text.append(np.array([word2idx[w] for w in sentence], dtype=np.int64))
        num_label.append(np.array([label2idx[l] for l in label], dtype=np.int64))
    return num_text, num_label

