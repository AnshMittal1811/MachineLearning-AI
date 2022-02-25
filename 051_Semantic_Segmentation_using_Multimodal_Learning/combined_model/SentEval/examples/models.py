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

import numpy as np
import io

import torch
from torch.autograd import Variable
import torch.nn as nn


"""
InferSent encoder
"""


class BLSTMEncoder(nn.Module):

    def __init__(self, config):
        super(BLSTMEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

    def get_sent_embedding_size(self):
        return self.enc_lstm_dim * 2 # Multiply by 2 since it's bidirectional

    def forward(self, sent, sent_len):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.use_cuda \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.use_cuda \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len))
            sent_len = sent_len.unsqueeze(1).cuda() if self.use_cuda \
                else sent_len.unsqueeze(1)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0].squeeze(0)

        return emb

    def set_glove_path(self, glove_path):
        self.glove_path = glove_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [s.split() if not tokenize else word_tokenize(s)
                     for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        return word_dict

    # Creates an embedding layer with the glove vecs for the CG's vocabulary
    def build_lookup_embedding(self, word_dict):
        # Vocab size +1 for the END token at index zero
        embedding_matrix = np.zeros((len(word_dict)+1, self.word_emb_dim,), dtype=np.float32)
        found_words = []
        with io.open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    embedding_matrix[int(word_dict[word])] = np.fromstring(vec, sep=' ')
                    found_words.append(word)

        print('Found {0}(/{1}) words with glove vectors'.format(len(found_words), len(word_dict)))
        if len(found_words) != len(word_dict):
            print("Missing words: ", [k for k in word_dict.keys() if k not in found_words])

        self.word_embeddings = torch.nn.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1],
            padding_idx=None, max_norm=None, norm_type=2,
            scale_grad_by_freq=False, sparse=False
        )
        if self.use_cuda:
            self.word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).cuda(), requires_grad=False)
        else:
            self.word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix), requires_grad=False)

    def get_glove_k(self, K):
        assert hasattr(self, 'glove_path'), 'warning : \
            you need to set_glove_path(glove_path)'
        # create word_vec with k first glove vectors
        k = 0
        word_vec = {}
        with io.open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in ['<s>', '</s>']:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k>K and all([w in word_vec for w in ['<s>', '</s>']]):
                    break
        return word_vec

    # The embedding matrix will be size_vocab x embedding_size
    # (This will later be multiplied by a 1-hot word vector of size 1 x size_vocab)
    def build_vocab_matrix(self, word_dict):
        assert hasattr(self, 'glove_path'), 'warning: \
            you need to set_glove_path(glove_path)'
        # Add +1 for the END token at index zero (word_dict is 1-indexed)
        self.vocab_size = len(word_dict)+1
        print('Vocab size : {0}'.format(self.vocab_size))
        self.build_lookup_embedding(word_dict)

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning: \
            you need to set_glove_path(glove_path)'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

    # build GloVe vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'glove_path'), 'warning: \
            you need to set_glove_path(glove_path)'
        self.word_vec = self.get_glove_k(K)
        print('Vocab size : {0}'.format(K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning: \
            you need to set_glove_path(glove_path)'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_glove(word_dict)
            self.word_vec.update(new_word_vec)
        print('New vocab size : {0} (added {1} words)'
              .format(len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def encode(self, sentences, lengths, gradients_enabled=True):
        self.input_batch = Variable(self.word_embeddings(sentences).data, requires_grad=gradients_enabled)
        embedded_batch = self.forward(self.input_batch, lengths)

        return embedded_batch
