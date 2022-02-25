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
Image-Caption Retrieval with COCO dataset
'''
from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import torch
from torch.autograd import Variable


try:
    import cPickle as pickle
except ImportError:
    import pickle

from .tools.ranking import ImageSentenceRankingPytorch

# Instantiated from senteval.py
class ImageCaptionRetrievalEval(object):
    def __init__(self, use_cuda, sentence_embedding_size, loss_function, task_path, seed=1111,
                     pretrained_senteval_path=None):
        logging.debug('***** Transfer task: Image Caption Retrieval *****\n\n')

        # Get captions and image features
        self.seed = seed
        # train, dev, test = self.loadFile(task_path)
        self.coco_data = self.loadFile(task_path)
        config_classifier = {'seed': self.seed, 'projdim': 1000, 'margin': 0.2}
        self.ranker = ImageSentenceRankingPytorch(use_cuda, sentence_embedding_size, loss_function, self.coco_data,
                                          config=config_classifier,
                                          pretrained_model_path=pretrained_senteval_path)


    def do_prepare(self, params, prepare):
        samples = None
        prepare(params, samples)

    def loadFile(self, fpath):
        # We need to be able to retreieve the coco_id from the id in the
        # data list here; this is needed to find the embedding for the
        # label-image during training (where the coco_id will be given)
        id_to_img_features = {}
        overlapping = 0
        max_id = 0
        for split in ['train', 'valid', 'test']:
            id_to_img_features[split] = {}
            with open(os.path.join(fpath, split + '.pkl')) as f:
                cocodata = pickle.load(f)
                #print("cocodata keys =", cocodata.keys())

            found_in_split = 0
            for imgkey in range(len(cocodata['features'])):
                # This assert makes sure we have all the captions - it's not
                # needed here, but lets us check the sanity of the data files
                #assert len(cocodata['image_to_caption_ids'][imgkey]) >= 5, \
                #       cocodata['image_to_caption_ids'][imgkey]

                # Store the image's CNN features filed under the original coco image id
                # (depending on the split and prefix, some have been relabeled in the SentEval datasets)
                [split_id, original_id] = cocodata['id_to_original_id'][imgkey].split('_') # example: 1_12345
                original_id = (int)(original_id)
                original_id = original_id - 1
                if split == 'train' and split_id == '0':
                    original_id += 40504
                if original_id in id_to_img_features[split]:
                    overlapping += 1
                id_to_img_features[split][original_id] = cocodata['features'][imgkey]
                found_in_split += 1
                max_id = max(original_id, max_id)
            #print("FOUND IN SPLIT", found_in_split)

        #print("MAX_ID", max_id)

        return id_to_img_features

    def t2i_stats(self, params, batcher, ordered_imgids, all_captions, all_lengths, split):
        # Embed captions batchwise to avoid OOM
        num_examples = len(all_lengths)
        all_embedded_captions = Variable(torch.zeros(num_examples, self.ranker.sentdim), volatile=True).cuda()
        embedded_index = 0
        start = 0
        end = min(start+params.batch_size, num_examples)
        while start < end:
            embedded_captions = batcher(params,
                                        torch.index_select(all_captions, 1, torch.LongTensor(range(start, end)).cuda()),
                                        all_lengths[start:end], gradients_enabled=False)
            if embedded_captions.dim() < 2:
                embedded_captions = embedded_captions.unsqueeze(0)

            # Store the current embedding batch in the complete embedding tensor
            next_index = embedded_index + embedded_captions.size(0)
            all_embedded_captions[embedded_index:next_index,:] = embedded_captions
            embedded_index = next_index

            start = end
            end = min(start+params.batch_size, num_examples)

        return self.ranker.t2i_stats(ordered_imgids, all_embedded_captions, split)

    # Takes a batch of sentences and their corresponding image_ids
    def caption_loss(self, params, batcher, correct_imgids, captions, lengths, split='train', contrastive_imgids=None,
                     loss_only=False):
        embedded_captions = batcher(params, captions, lengths, gradients_enabled=(not loss_only))
        if embedded_captions.dim() < 2:
            embedded_captions = embedded_captions.unsqueeze(0)

        loss = self.ranker.caption_loss(correct_imgids, embedded_captions, split, contrastive_imgids, loss_only)

        if loss_only:
            return loss
        else:
            # To get the grads wrt the sampled inputs, take the gradient wrt the
            # inputs to the encoder multiplied by the embedding matrix (because the
            # grad wrt x in x * emb = emb)
            # TODO Prepare the right-hand matrix once on creation, based on the batch size
            sizes = [params.infersent.input_batch.grad.size(0), params.infersent.word_embeddings.weight.transpose(0,1).size(0), params.infersent.word_embeddings.weight.transpose(0,1).size(1)]
            return loss, torch.bmm(params.infersent.input_batch.grad, params.infersent.word_embeddings.weight.transpose(0,1).unsqueeze(0).expand(sizes[0], sizes[1], sizes[2])).detach().data

    def run(self, params, batcher):
        coco_embed = {'train': {'sentfeat': [], 'imgfeat': []},
                      'dev': {'sentfeat': [], 'imgfeat': []},
                      'test': {'sentfeat': [], 'imgfeat': []}}

        for key in self.coco_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            self.coco_data[key]['sent'] = np.array(self.coco_data[key]['sent'])
            self.coco_data[key]['sent'], idx_sort = np.sort(self.coco_data[key]['sent']), np.argsort(self.coco_data[key]['sent'])
            idx_unsort = np.argsort(idx_sort)

            coco_embed[key]['X'] = []
            nsent = len(self.coco_data[key]['sent'])
            for ii in range(0, nsent, params.batch_size):
                batch = self.coco_data[key]['sent'][ii:ii + params.batch_size]
                embeddings = batcher(params, batch)
                coco_embed[key]['sentfeat'].append(embeddings)
            coco_embed[key]['sentfeat'] = np.vstack(coco_embed[key]['sentfeat'])[idx_unsort]
            coco_embed[key]['imgfeat'] = np.array(self.coco_data[key]['imgfeat'])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'seed': self.seed, 'projdim': 1000, 'margin': 0.2}
        clf = ImageSentenceRankingPytorch(train=coco_embed['train'],
                                          valid=coco_embed['dev'],
                                          test=coco_embed['test'],
                                          config=config_classifier)

        bestdevscore, r1_i2t, r5_i2t, r10_i2t, medr_i2t, \
            r1_t2i, r5_t2i, r10_t2i, medr_t2i = clf.run()

        logging.debug("\nTest scores | Image to text: \
            {0}, {1}, {2}, {3}".format(r1_i2t, r5_i2t, r10_i2t, medr_i2t))
        logging.debug("Test scores | Text to image: \
            {0}, {1}, {2}, {3}\n".format(r1_t2i, r5_t2i, r10_t2i, medr_t2i))

        return {'devacc': bestdevscore,
                'acc': [(r1_i2t, r5_i2t, r10_i2t, medr_i2t),
                        (r1_t2i, r5_t2i, r10_t2i, medr_t2i)],
                'ndev': len(coco_embed['dev']['sentfeat']),
                'ntest': len(coco_embed['test']['sentfeat'])}
