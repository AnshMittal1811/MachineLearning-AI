import itertools
from collections import defaultdict
from spacy.lang.en import English
from tqdm import tqdm
nlp = English()
tokenizer = nlp.tokenizer

WIKI_DATA_PATH = "data/raw/wikipedia/wiki.tsv"
OPTED_DICT_PATH = "heuristic-data-collection/dictionary/opted_dict.tsv"

def bigrams(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))

candidate_words = defaultdict(int)

# add dictionary words directly to final wordlist
with open(OPTED_DICT_PATH, 'r') as f:
    for index, line in tqdm(enumerate(f)):
        word = line.split('\t')[1]
        candidate_words[word] = 1

paragraphs = []
titles = []
with open(WIKI_DATA_PATH, 'r') as f:
    for index, line in tqdm(enumerate(f)):
        paragraphs.append(line.split('\t')[1])
        titles.append(line.split('\t')[2])

# count all unigrams and bigrams from title and paragraph as word candidates
for tokens in tqdm(tokenizer.pipe(titles, batch_size=10000)):
    tokens = [t.text for t in tokens]
    for token in tokens:
        candidate_words[token] += 1
    for bigram in bigrams(tokens):
        candidate_words[bigram] += 1

for tokens in tqdm(tokenizer.pipe(paragraphs, batch_size=10000)):
    tokens = [t.text for t in tokens]
    for token in tokens:
        candidate_words[token] += 1
    for bigram in bigrams(tokens):
        candidate_words[bigram] += 1

import pickle
with open('wiki_ngrams.pkl', 'wb') as f:
    pickle.dump(candidate_words, f)
