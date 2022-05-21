#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to generate distantly supervised training data.

Using Wikipedia and available QA datasets, we search for a paragraph
that can be used as a supporting context.
"""

import argparse
import uuid
import heapq
import logging
import regex as re
import os
import json
import csv
import random

from functools import partial
from collections import Counter
from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize

from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

from drqa import tokenizers
from drqa import retriever
from drqa.retriever import utils

logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Fetch text, tokenize + annotate
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, tokenizer_opts, db_class=None, db_opts=None, db_path=None):
    global PROCESS_TOK, PROCESS_DB
    #print(tokenizer_opts)
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)

    db_opts = {"db_path": "data/processed/wikipedia/docs.db"}
    #if db_path:
    #    db_opts.update({"db_path": db_path})
    # optionally open a db connection
    if db_class:
        PROCESS_DB = db_class(**db_opts)
        Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def nltk_entity_groups(text):
    """Return all contiguous NER tagged chunks by NLTK."""
    return None


# ------------------------------------------------------------------------------
# Find answer candidates.
# ------------------------------------------------------------------------------

def check_contains(words, term):
    """ search for terms in word iteratively
    """
    joined = ""
    for w in words:
        joined += words.pop(0)
        if len(joined) > len(term):
            return False
        if joined == term:
            return True
    return False

def find_answer(paragraph, q_tokens, answer, opts):
    """Return the best matching answer offsets from a paragraph.

    The paragraph is skipped if:
    * It is too long or short.
    * It doesn't contain the answer at all.
    * It doesn't contain named entities found in the question.
    * The answer context match score is too low.
      - This is the unigram + bigram overlap within +/- window_sz.
    """
    # Length check
    if len(paragraph) > opts['char_max'] or len(paragraph) < opts['char_min']:
        return False, 0

    # Answer check
    answer = [answer]
    answers = {a for a in answer if a.lower().replace(" ", "") in paragraph.replace(" ", "").lower()}
    if len(answers) == 0:
        return False, 0

    # Search...
    p_tokens = tokenize_text(paragraph)
    p_words = p_tokens.words(uncased=True)
    q_tokens = q_tokens[0]
    q_grams = Counter(q_tokens.ngrams(
        n=2, uncased=True, filter_fn=utils.filter_ngram
    ))

    best_score = 0
    best_ex = None
    #print(answers)
    for ans in answers:
        try:
            a_words = tokenize_text(ans).words(uncased=True)
        except RuntimeError:
            logger.warn('Failed to tokenize answer: %s' % ans)
            continue
        for idx in range(len(p_words)):
            #if sum(p_words[idx:idx + 5], [])[:len(a_words[0])] == a_words[0]: # we arbitrarily loosen the constraint to 5 word windows
            if check_contains(p_words[idx: idx + len(a_words[0])], a_words[0]):
                # Overlap check
                w_s = max(idx - opts['window_sz'], 0)
                w_e = min(idx + opts['window_sz'] + len(a_words), len(p_words))
                w_tokens = p_tokens.slice(w_s, w_e)
                w_grams = Counter(w_tokens.ngrams(
                    n=2, uncased=True, filter_fn=utils.filter_ngram
                ))
                score = sum((w_grams & q_grams).values())
                if score > best_score:
                    # Success! Set new score + formatted example
                    best_score = score
                    best_ex = {
                        'id': uuid.uuid4().hex,
                        'question': q_tokens.words(),
                        'document': p_tokens.words(),
                        #'offsets': p_tokens.offsets(),
                        #'answers': [(idx, idx + len(a_words) - 1)],
                        #'qlemma': q_tokens.lemmas(),
                        #'lemma': p_tokens.lemmas(),
                        #'pos': p_tokens.pos(),
                        #'ner': p_tokens.entities(),
                    }
    if best_score >= opts['match_threshold']:
        return True, best_score
    else:
        return False, 0    



def search_docs(inputs, max_ex=5, opts=None):
    """Given a set of document ids (returned by ranking for a question), search
    for top N best matching (by heuristic) paragraphs that contain the answer.
    """
    if not opts:
        raise RuntimeError('Options dict must be supplied.')

    current_id, doc_ids, q_tokens, answer = inputs
    examples = []
    negative_examples = []
    for i, doc_id in enumerate(doc_ids):
        #for j, paragraph in enumerate(re.split(r'\n+', fetch_text(doc_id))):
        paragraph = fetch_text(doc_id)
        found, score = find_answer(paragraph, q_tokens, answer, opts)
        if found:
            # Reverse ranking, giving priority to early docs + paragraphs
            j = i
            score = (score, -i, -j, random.random())
            if len(examples) < max_ex:
                heapq.heappush(examples, (score, doc_id))
            else:
                heapq.heappushpop(examples, (score, doc_id))
        else:
            negative_examples.append(doc_id)
    return current_id, [e[1] for e in examples], negative_examples


def process(questions, answers, outfile, opts):
    """Generate examples for all questions."""
    logger.info('Processing %d question answer pairs...' % len(questions))
    logger.info('Will save to %s.dstrain and %s.dsdev' % (outfile, outfile))

    # Load ranker
    ranker = opts['ranker_class'](strict=False, tfidf_path="/home/ericwallace/albertxu/xword/data/processed/wikipedia/tfidf/docs-tfidf-ngram=2-hash=16777216-tokenizer=spacy.npz")
    logger.info('Ranking documents (top %d per question)...' % opts['n_docs'])
    ranked = ranker.batch_closest_docs(questions, k=opts['n_docs'])
    #print(ranked)
    ranked = [r[0] for r in ranked]

    # Start pool of tokenizers with ner enabled
    workers = Pool(opts['workers'], initializer=init,
                   initargs=(opts['tokenizer_class'], {'annotators': {'ner'}}))

    logger.info('Pre-tokenizing questions...')
    q_tokens = workers.map(tokenize_text, questions)
    q_ner = workers.map(nltk_entity_groups, questions)
    q_tokens = list(zip(q_tokens, q_ner))
    workers.close()
    workers.join()

    # Start pool of simple tokenizers + db connections
    workers = Pool(opts['workers'], initializer=init,
                   initargs=(opts['tokenizer_class'], {},
                             opts['db_class'], {}))

    logger.info('Searching documents...')
    cnt = 0
    inputs = [(i, ranked[i], q_tokens[i], answers[i]) for i in range(len(ranked))]
    #print(inputs)
    search_fn = partial(search_docs, max_ex=opts['max_ex'], opts=opts['search'])
    with open(outfile + '.dstrain', 'w') as f_train, \
         open(outfile + '.dsdev', 'w') as f_dev:
        train_writer = csv.writer(f_train, delimiter="\t")
        dev_wroter = csv.writer(f_dev, delimiter="\t")
        for ex in workers.imap_unordered(search_fn, inputs):
            #print(ex)
            ex_proc = [ex[0], "" if not ex[1] else ex[1][0], "" if not ex[2] else ex[2][0]]
            cnt += 1
            f = dev_writer if random.random() < opts['dev_split'] else train_writer
            f.writerow(ex_proc)
            if cnt % 1000 == 0:
                logging.info('%d results so far...' % cnt)
    workers.close()
    workers.join()
    logging.info('Finished. Total = %d' % cnt)


# ------------------------------------------------------------------------------
# Main & commandline options
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('data_name', type=str, help='Dataset name')
    parser.add_argument('out_dir', type=str, help='Output directory')

    dataset = parser.add_argument_group('Dataset')
    dataset.add_argument('--regex', action='store_true',
                         help='Flag if answers are expressed as regexps')
    dataset.add_argument('--dev-split', type=float, default=0,
                         help='Hold out for ds dev set (0.X)')

    search = parser.add_argument_group('Search Heuristic')
    search.add_argument('--match-threshold', type=int, default=1,
                        help='Minimum context overlap with question')
    search.add_argument('--char-max', type=int, default=1500,
                        help='Maximum allowed context length')
    search.add_argument('--char-min', type=int, default=25,
                        help='Minimum allowed context length')
    search.add_argument('--window-sz', type=int, default=20,
                        help='Use context on +/- window_sz for overlap measure')

    general = parser.add_argument_group('General')
    general.add_argument('--max-ex', type=int, default=1,
                         help='Maximum matches generated per question')
    general.add_argument('--n-docs', type=int, default=100,
                         help='Number of docs retrieved per question')
    general.add_argument('--tokenizer', type=str, default='spacy')
    general.add_argument('--ranker', type=str, default='tfidf')
    general.add_argument('--db', type=str, default='sqlite')
    general.add_argument('--workers', type=int, default=cpu_count())
    args = parser.parse_args()

    # Logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Read dataset
    dataset = os.path.join(args.data_dir, args.data_name)
    questions = []
    answers = []
    #for line in open(dataset):
    #    data = json.loads(line)
    #    question = data['question']
    #    answer = data['answer']
    #
    #    # Make sure the regex compiles
    #    if args.regex:
    #        try:
    #            re.compile(answer[0])
    #        except BaseException:
    #            logger.warning('Regex failed to compile: %s' % answer)
    #            continue
    with open(dataset) as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for question, answer in reader:
            questions.append(question)
            answers.append(answer[2:-2]) # strip off [' and ']

    # Get classes
    ranker_class = retriever.get_class(args.ranker)
    db_class = retriever.get_class(args.db)
    tokenizer_class = tokenizers.get_class(args.tokenizer)

    # Form options
    search_keys = ('regex', 'match_threshold', 'char_max',
                   'char_min', 'window_sz')
    
    opts = {
        'ranker_class': retriever.get_class(args.ranker),
        'tokenizer_class': tokenizers.get_class(args.tokenizer),
        'db_class': retriever.get_class(args.db),
        'search': {k: vars(args)[k] for k in search_keys},
    }
    opts.update(vars(args))

    # Process!
    outname = os.path.splitext(args.data_name)[0]
    outfile = os.path.join(args.out_dir, outname)
    process(questions, answers, outfile, opts)
