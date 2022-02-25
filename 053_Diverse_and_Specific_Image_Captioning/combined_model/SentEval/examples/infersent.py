# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file is originally from:
# https://github.com/facebookresearch/SentEval
# It contains changes relating to the paper 'Generating Diverse and Meaningful
# Captions: Unsupervised Specificity Optimization for Image Captioning (Lindh
# et al., 2018)'. For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning
# The original copyright message is preserved below. This should not be seen as
# an endorsement from Facebook relating to this modified version of the code.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
from .exutil import dotdict
import logging


# Set PATHs
GLOVE_PATH = os.path.join(os.getcwd(), 'glove/glove.42B.300d.txt')
PATH_SENTEVAL = '../'
PATH_TO_DATA = os.path.join(os.getcwd(), 'SentEval/data/senteval_data')
MODEL_PATH = os.path.join(os.getcwd(), 'SentEval/examples/infersent.allnli.pickle')

assert os.path.isfile(MODEL_PATH) and os.path.isfile(GLOVE_PATH), \
    'Set MODEL and GloVe PATHs'

sys.path.insert(0, PATH_SENTEVAL)
from .. import senteval


def prepare(params, samples):
    # No longer used for our training
    pass


def batcher(params, sentences, lengths, gradients_enabled=True):
    # batch contains list of words
    embeddings = params.infersent.encode(sentences, lengths, gradients_enabled)

    return embeddings


# Load SentEval with InferSent from another script
def load_senteval(vocab_dict, batch_size, loss_function, use_cuda=True, pretrained_senteval_path=None):
    # define senteval params
    params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                               'seed': 1111, 'kfold': 5, 'batch_size': batch_size})

    # We need to remap all cuda ids to 0 since their file has stored their internal cuda ids
    params_senteval.infersent = torch.load(MODEL_PATH, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0'})
    params_senteval.infersent.use_cuda = use_cuda
    params_senteval.infersent.set_glove_path(GLOVE_PATH)
    params_senteval.infersent.build_vocab_matrix(vocab_dict)

    se = senteval.SentEval(use_cuda, params_senteval.infersent.get_sent_embedding_size(), loss_function,
                           params_senteval, batcher, prepare, pretrained_senteval_path=pretrained_senteval_path)

    return se


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC',
                  'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']

# define senteval params
params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                           'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load model
    params_senteval.infersent = torch.load(MODEL_PATH, map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
    params_senteval.infersent.set_glove_path(GLOVE_PATH)

    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
