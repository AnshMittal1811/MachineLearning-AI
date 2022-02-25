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

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from .rank import ImageCaptionRetrievalEval


class SentEval(object):
    def __init__(self, use_cuda, sentence_embedding_size, loss_function, params,
                 batcher, prepare=None, pretrained_senteval_path=None):
        # setting default parameters
        params.usepytorch = True if 'usepytorch' not in params else \
            params.usepytorch
        params.classifier = 'LogReg' if 'classifier' not in params else \
            params.classifier
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.batch_size = 128 if 'batch_size' not in params else \
            params.batch_size
        params.seed = 1111 if 'seed' not in params else params.seed
        params.kfold = 5 if 'kfold' not in params else params.kfold
        self.params = params

        self.batcher = batcher
        if prepare:
            self.prepare = prepare
        else:
            self.prepare = lambda x, y: None

        # sanity check
        assert params.classifier in ['LogReg', 'MLP']
        if params.classifier == 'MLP':
            assert params.nhid > 0, 'When using an MLP, \
                you need to set params.nhid>0'
        if not params.usepytorch and params.classifier == 'MLP':
            assert False, 'No MLP implemented in scikit-learn'

        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16']

        # Prepare the IR evaluator for our IR loss calculations
        self.evaluation = ImageCaptionRetrievalEval(use_cuda, sentence_embedding_size, loss_function,
                                    self.params.task_path + '/COCO', seed=self.params.seed,
                                    pretrained_senteval_path=pretrained_senteval_path)
        self.params.current_task = 'ImageCaptionRetrieval'
        # The modified version of prepare will load the CG's vocabulary
        self.evaluation.do_prepare(self.params, self.prepare)

    def enable_learning(self, enable=True):
        for p in self.evaluation.ranker.model.parameters():
            p.volatile = not enable
        for p in self.params.infersent.parameters():
            p.volatile = not enable

    def t2i_stats(self, ordered_imgids, all_captions, all_lengths, split):
        return self.evaluation.t2i_stats(self.params, self.batcher, ordered_imgids, all_captions, all_lengths, split)

    def caption_loss(self, image_ids, sentences, lengths, split='train', contrastive_imgids=None, loss_only=False):
        loss = self.evaluation.caption_loss(self.params, self.batcher, image_ids, sentences, lengths, split,
                                            contrastive_imgids, loss_only)

        return loss
