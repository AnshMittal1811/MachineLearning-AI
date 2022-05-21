import sys
import utils

import os
import re
from tqdm import tqdm
import spacy
from ftfy import fix_text
import random
import pickle
from string import ascii_uppercase

SEG_DATA_DIR = "data/"

def make_data(str_lst):
    nlp = spacy.load("en_core_web_sm")

    clues = []
    sentences = []
    for s in tqdm(str_lst):
        orig = fix_text(s.split(",")[0].split("(")[0])
        orig = re.sub("^ | $", "", orig)
        no_spaces = re.sub(" ", "", orig)
        clue = no_spaces.upper()
        if re.search("[^A-Z]", clue):
            continue
        if 3 <= len(clue) <= 15:
            clues.append(clue)
            sentences.append(orig)

    return list(zip(clues, sentences))

def collapse_bigrams(dictionary):
    ngrams = []
    for val in dictionary.keys():
        if type(val) == tuple:
            first = re.sub(" ", "", val[0])
            second = re.sub(" ", "", val[1])
            bigram = first + " " + second
            ngrams.append(bigram)
        else:
           ngrams.append(val)
    return ngrams

if __name__ == "__main__":
    TRAIN_PATH = SEG_DATA_DIR + "processed/train.txt"
    VALID_PATH = SEG_DATA_DIR + "processed/valid.txt"
    OTHER_TEXT = list(map(lambda p: os.path.join(SEG_DATA_DIR + "raw/", p), os.listdir(SEG_DATA_DIR + "raw/")))
    
    titles = []
    for fname in OTHER_TEXT:
        titles += utils.read_text(fname)

    pickle_fn = SEG_DATA_DIR + "wiki_ngrams.pkl"
    print("Reading pickle file...")
    wiki_ngrams = pickle.load(open(pickle_fn, "rb"))
    print("Pickle file size: {}".format(len(wiki_ngrams)))
    print("Sorting dictionary...")
    wiki_ngrams = {k:v for k, v in sorted(wiki_ngrams.items(), key=lambda val: val[1], reverse=True) if v >= 25}
    print("Pickle file size after truncation: {}".format(len(wiki_ngrams)))
    wiki_ngrams_tokens = collapse_bigrams(wiki_ngrams)

    data_lst = list(set((titles + wiki_ngrams_tokens)))
    data_lst = make_data(data_lst)
    random.shuffle(data_lst)

    num_train = int(1 * len(data_lst))
    train_set, valid_set = data_lst[:num_train], data_lst[num_train:]
    print("Train Size: {}".format(train_set))
    print("Valid Size: {}".format(valid_set))
    utils.write_csv(train_set, TRAIN_PATH)
    utils.write_csv(valid_set, VALID_PATH)
