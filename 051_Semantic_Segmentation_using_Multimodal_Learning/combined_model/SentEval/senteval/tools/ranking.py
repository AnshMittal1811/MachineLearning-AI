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

"""
Image Annotation/Search for COCO with Pytorch
"""
from __future__ import absolute_import, division, unicode_literals

import logging
import copy
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from neuraltalk2_pytorch.misc import utils

# Divide similarity calculations into batches to preserve GPU memory
_SIMILARITY_BATCH_SIZE = 100


class COCOProjNet(nn.Module):
    def __init__(self, config):
        super(COCOProjNet, self).__init__()
        self.imgdim = config['imgdim']
        self.sentdim = config['sentdim']
        self.projdim = config['projdim']
        self.use_cuda = config['use_cuda']
        self.imgproj = nn.Sequential(
                        nn.Linear(self.imgdim, self.projdim),
                        )
        self.sentproj = nn.Sequential(
                        nn.Linear(self.sentdim, self.projdim),
                        )

    def forward(self, imgproj, sentproj, imgcproj, sentcproj):
        # imgc : (bsize, ncontrast, imgdim)
        # sentc : (bsize, ncontrast, sentdim)
        # img : (bsize, imgdim)
        # sent : (bsize, sentdim)

        # (bsize*ncontrast, projdim)

        anchor1 = torch.sum((imgproj*sentproj), 1)
        anchor2 = torch.sum((sentproj*imgproj), 1)
        img_sentc = torch.sum((imgproj*sentcproj), 1)
        sent_imgc = torch.sum((sentproj*imgcproj), 1)

        # (bsize*ncontrast)
        return anchor1, anchor2, img_sentc, sent_imgc

    def proj_sentence(self, sent):
        output = self.sentproj(sent)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output # (bsize, projdim)

    def proj_image(self, img):
        output = self.imgproj(img)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output # (bsize, projdim)


# Instantiated in rank.py
class ImageSentenceRankingPytorch(object):
    # Image Sentence Ranking on COCO with Pytorch
    def __init__(self, use_cuda, sentence_embedding_size, loss_function, id_to_img_features,
                 config, pretrained_model_path=None):
        self.use_cuda = use_cuda
        assert (loss_function == 'cosine_similarity' or loss_function == 'direct_similarity' or
                loss_function == 'pairwise_cosine' or loss_function == 'pairwise_similarity'), loss_function
        self.loss_function = loss_function

        # fix seed
        self.seed = config['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.seed)

        self.imgdim = len(id_to_img_features['train'][next(iter(id_to_img_features['train']))])
        # The value in sentence_embedding_size comes from the loaded infersent model
        self.sentdim = sentence_embedding_size #len(train['sentfeat'][0])
        self.projdim = config['projdim']
        self.margin = config['margin']

        self.batch_size = 128
        self.ncontrast = 30
        self.maxepoch = 20
        self.early_stop = True

        config_model = {'imgdim': self.imgdim,'sentdim': self.sentdim,
                        'projdim': self.projdim, 'use_cuda': self.use_cuda}

        if self.use_cuda:
            self.model = COCOProjNet(config_model).cuda()
        else:
            self.model = COCOProjNet(config_model)

        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path)
            print("STATE_DICT", state_dict)
            self.model.load_state_dict(state_dict)

        # Prepare by projecting all images since this data will be constant
        self.project_all_images(id_to_img_features)

        self.cos = nn.CosineSimilarity()

    # Project all images since their embeddings won't change
    def project_all_images(self, id_to_img_features):
        # First, disable gradients for the model's image projection since this is frozen
        for p in self.model.imgproj.parameters():
            p.requires_grad = False

        # Store all projected images as one matrix per split
        self.projected_images = {}
        # Link the image ids to their indices in the split-matrices
        self.image_id_to_index = {}

        # Project all images and keep them linked to their img_ids
        for split in ['train', 'valid', 'test']:
            projected_split_images = list()
            image_index = 0
            self.image_id_to_index[split] = {}
            for img_id in id_to_img_features[split]:
                # The unsqueeze is needed to add the batch dimension since we're using singles here
                if self.use_cuda:
                    img_tensor = torch.from_numpy(id_to_img_features[split][img_id]).unsqueeze(0).cuda()
                else:
                    img_tensor = torch.from_numpy(id_to_img_features[split][img_id]).unsqueeze(0)

                projected_split_images.append(self.model.proj_image(
                    Variable(img_tensor, requires_grad=False)))
                self.image_id_to_index[split][img_id] = image_index
                image_index += 1
            # Store as matrix with dims = projection_size x num_images (to avoid big transpose when
            # calculating similarity to all images of a split)
            self.projected_images[split] = torch.cat([img.squeeze().unsqueeze(1) for img in projected_split_images], 1)

    def prepare_data(self, imgs):
        if self.use_cuda:
            return torch.FloatTensor(imgs).cuda()
        else:
            return torch.FloatTensor(imgs)

    def run(self):
        self.nepoch = 0
        bestdevscore = -1
        early_stop_count = 0
        stop_train = False

        # Preparing data
        logging.info('prepare data')
        trainTxt, trainImg, devTxt, devImg, testTxt, testImg = \
            self.prepare_data(self.train['sentfeat'], self.train['imgfeat'],
                              self.valid['sentfeat'], self.valid['imgfeat'],
                              self.test['sentfeat'], self.test['imgfeat'])

        # Training
        while not stop_train and self.nepoch <= self.maxepoch:
            logging.info('start epoch')
            self.trainepoch(trainTxt, trainImg, devTxt, devImg, nepoches=1)
            logging.info('Epoch {0} finished'.format(self.nepoch))

            results = {'i2t': {'r1': 0, 'r5': 0, 'r10': 0, 'medr': 0},
                       't2i': {'r1': 0, 'r5': 0, 'r10': 0, 'medr': 0},
                       'dev': bestdevscore}
            score = 0
            for i in range(5):
                devTxt_i = devTxt[i*5000:(i+1)*5000]
                devImg_i = devImg[i*5000:(i+1)*5000]
                # Compute dev ranks img2txt
                r1_i2t, r5_i2t, r10_i2t, medr_i2t = self.i2t(devImg_i,
                                                             devTxt_i)
                results['i2t']['r1'] += r1_i2t / 5
                results['i2t']['r5'] += r5_i2t / 5
                results['i2t']['r10'] += r10_i2t / 5
                results['i2t']['medr'] += medr_i2t / 5
                logging.info("Image to text: {0}, {1}, {2}, {3}"
                             .format(r1_i2t, r5_i2t, r10_i2t, medr_i2t))
                # Compute dev ranks txt2img
                r1_t2i, r5_t2i, r10_t2i, medr_t2i = self.t2i(devImg_i,
                                                             devTxt_i)
                results['t2i']['r1'] += r1_t2i / 5
                results['t2i']['r5'] += r5_t2i / 5
                results['t2i']['r10'] += r10_t2i / 5
                results['t2i']['medr'] += medr_t2i / 5
                logging.info("Text to Image: {0}, {1}, {2}, {3}"
                             .format(r1_t2i, r5_t2i, r10_t2i, medr_t2i))
                score += (r1_i2t + r5_i2t + r10_i2t +
                          r1_t2i + r5_t2i + r10_t2i) / 5

            logging.info("Dev mean Text to Image: {0}, {1}, {2}, {3}".format(
                        results['t2i']['r1'], results['t2i']['r5'],
                        results['t2i']['r10'], results['t2i']['medr']))
            logging.info("Dev mean Image to text: {0}, {1}, {2}, {3}".format(
                        results['i2t']['r1'], results['i2t']['r5'],
                        results['i2t']['r10'], results['i2t']['medr']))

            # early stop on Pearson
            if score > bestdevscore:
                bestdevscore = score
                bestmodel = copy.deepcopy(self.model)
            elif self.early_stop:
                if early_stop_count >= 3:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel

        # Compute test for the 5 splits
        results = {'i2t': {'r1': 0, 'r5': 0, 'r10': 0, 'medr': 0},
                   't2i': {'r1': 0, 'r5': 0, 'r10': 0, 'medr': 0},
                   'dev': bestdevscore}
        for i in range(5):
            testTxt_i = testTxt[i*5000:(i+1)*5000]
            testImg_i = testImg[i*5000:(i+1)*5000]
            # Compute test ranks img2txt
            r1_i2t, r5_i2t, r10_i2t, medr_i2t = self.i2t(testImg_i, testTxt_i)
            results['i2t']['r1'] += r1_i2t / 5
            results['i2t']['r5'] += r5_i2t / 5
            results['i2t']['r10'] += r10_i2t / 5
            results['i2t']['medr'] += medr_i2t / 5
            # Compute test ranks txt2img
            r1_t2i, r5_t2i, r10_t2i, medr_t2i = self.t2i(testImg_i, testTxt_i)
            results['t2i']['r1'] += r1_t2i / 5
            results['t2i']['r5'] += r5_t2i / 5
            results['t2i']['r10'] += r10_t2i / 5
            results['t2i']['medr'] += medr_t2i / 5

        return bestdevscore, results['i2t']['r1'], results['i2t']['r5'], \
                             results['i2t']['r10'], results['i2t']['medr'], \
                             results['t2i']['r1'], results['t2i']['r5'], \
                             results['t2i']['r10'], results['t2i']['medr']

    # This is the SentEval training function, not used for the specificity optimization of the NLG model
    def trainepoch(self, trainTxt, trainImg, devTxt, devImg, nepoches=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + nepoches):
            permutation = list(np.random.permutation(len(trainTxt)))
            all_costs = []
            for i in range(0, len(trainTxt), self.batch_size):
                # forward
                if i % (self.batch_size*500) == 0 and i > 0:
                    logging.info('samples : {0}'.format(i))
                    r1_i2t, r5_i2t, r10_i2t, medr_i2t = self.i2t(devImg,
                                                                 devTxt)
                    logging.info("Image to text: {0}, {1}, {2}, {3}".format(
                        r1_i2t, r5_i2t, r10_i2t, medr_i2t))
                    # Compute test ranks txt2img
                    r1_t2i, r5_t2i, r10_t2i, medr_t2i = self.t2i(devImg,
                                                                 devTxt)
                    logging.info("Text to Image: {0}, {1}, {2}, {3}".format(
                        r1_t2i, r5_t2i, r10_t2i, medr_t2i))
                idx = torch.LongTensor(permutation[i:i + self.batch_size])
                imgbatch = Variable(trainImg.index_select(0, idx)).cuda()
                sentbatch = Variable(trainTxt.index_select(0, idx)).cuda()

                idximgc = np.random.choice(permutation[:i] +
                                           permutation[i + self.batch_size:],
                                           self.ncontrast*idx.size(0))
                idxsentc = np.random.choice(permutation[:i] +
                                            permutation[i + self.batch_size:],
                                            self.ncontrast*idx.size(0))
                idximgc = torch.LongTensor(idximgc)
                idxsentc = torch.LongTensor(idxsentc)
                # Get indexes for contrastive images and sentences
                imgcbatch = Variable(trainImg.index_select(0, idximgc)).view(
                    -1, self.ncontrast, self.imgdim).cuda()
                sentcbatch = Variable(trainTxt.index_select(0, idxsentc)).view(
                    -1, self.ncontrast, self.sentdim).cuda()

                anchor1, anchor2, img_sentc, sent_imgc = self.model(
                    imgbatch, sentbatch, imgcbatch, sentcbatch)
                # loss
                loss = self.loss_fn(anchor1, anchor2, img_sentc, sent_imgc)
                all_costs.append(loss.data[0])
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += nepoches

    def image_id_to_features(self, image_ids, split):
        image_indices = torch.LongTensor([self.image_id_to_index[split][img_id] for img_id in image_ids]).cuda()
        return self.projected_images[split].data.index_select(1, image_indices)

    # DP - Dot Product similarity loss
    def loss_direct_similarity(self, correct_imgids, captions, split):
        # Get image and caption projections
        imgfeats_tensor = self.image_id_to_features(correct_imgids, split)  # embedding_size x batch_size
        images = Variable(imgfeats_tensor.transpose(0, 1).unsqueeze(2), requires_grad=False)  # batch_size x embedding_size x 1
        sent_embed = self.model.proj_sentence(captions).unsqueeze(1)  # batch_size x 1 x embedding_size

        scores = torch.bmm(sent_embed, images)
        return -(scores).mean()

    # CDP - Contrastive Dot Product similarity loss
    def loss_pairwise_similarity(self, correct_imgids, captions, split, contrastive_imgids):
        imgfeats_tensor = self.image_id_to_features(correct_imgids, split)  # embedding_size x batch_size
        images = Variable(imgfeats_tensor.transpose(0, 1).unsqueeze(2), requires_grad=False)  # batch_size x embedding_size x 1

        contrastive_imgfeats_tensor = self.image_id_to_features(contrastive_imgids, split)  # embedding_size x batch_size
        contrastive_images = Variable(contrastive_imgfeats_tensor.transpose(0, 1).unsqueeze(2), requires_grad=False)  # batch_size x embedding_size x 1

        sent_embed = self.model.proj_sentence(captions).unsqueeze(1) # batch_size x 1 x embedding_size

        loss = torch.clamp(torch.bmm(sent_embed, contrastive_images) - torch.bmm(sent_embed, images), min=0).mean()

        return loss

    # Cos - The cosine loss between the correct image and the generated caption
    def loss_cosine_similarity(self, correct_imgids, captions, split):
        # Get image and caption projections
        imgfeats_tensor = self.image_id_to_features(correct_imgids, split) # embedding_size x batch_size
        images = Variable(imgfeats_tensor.transpose(0, 1), requires_grad=False) # batch_size x embedding_size
        sent_embed = self.model.proj_sentence(captions) # batch_size x embedding_size

        scores = self.cos(sent_embed, images)
        return -(scores).mean()

    # CCos - Contrastive Cosine similarity loss
    def loss_pairwise_cosine(self, correct_imgids, captions, split, contrastive_imgids):
        imgfeats_tensor = self.image_id_to_features(correct_imgids, split)  # embedding_size x batch_size
        images = Variable(imgfeats_tensor.transpose(0, 1), requires_grad=False)  # batch_size x embedding_size

        contrastive_imgfeats_tensor = self.image_id_to_features(contrastive_imgids, split)  # embedding_size x batch_size
        contrastive_images = Variable(contrastive_imgfeats_tensor.transpose(0, 1), requires_grad=False)  # batch_size x embedding_size

        sent_embed = self.model.proj_sentence(captions) # batch_size x embedding_size

        loss = torch.clamp(self.cos(sent_embed, contrastive_images) - self.cos(sent_embed, images), min=0).mean()

        return loss

    # This is called directly from rank.py's caption_loss() instead of run()
    def caption_loss(self, correct_imgids, captions, split='train', contrastive_imgids=None, loss_only=False):
        if not loss_only:
            # Reset the previous gradient calculations
            self.model.zero_grad()

        # Compute scores
        if self.loss_function == 'cosine_similarity':
            loss = self.loss_cosine_similarity(correct_imgids, captions, split)
        elif self.loss_function == 'direct_similarity':
            loss = self.loss_direct_similarity(correct_imgids, captions, split)
        elif self.loss_function == 'pairwise_cosine':
            loss = self.loss_pairwise_cosine(correct_imgids, captions, split, contrastive_imgids)
        elif self.loss_function == 'pairwise_similarity':
            loss = self.loss_pairwise_similarity(correct_imgids, captions, split, contrastive_imgids)

        if not loss_only:
            loss.backward()

        return loss.detach().data

    # Used for validation and testing of the specificity optimization
    def t2i_stats(self, ordered_imgids, all_captions, split):
        # Get image and caption projections
        embedded_images = Variable(self.image_id_to_features(ordered_imgids, split), volatile=True)
        embedded_captions = self.model.proj_sentence(all_captions)
        num_examples = len(embedded_captions)

        # Compute similarity scores on the GPU and then move the results to the CPU
        score_table = torch.mm(embedded_captions, embedded_images).data.cpu().numpy()

        # Calculate ranks and relative cosine distance
        ranks = np.zeros(num_examples)
        scores = np.zeros(num_examples)
        relative_scores_at5 = np.zeros(num_examples)
        relative_scores_at10 = np.zeros(num_examples)
        for i_example in range(num_examples):
            # Store the scores along the diagonal (score of correct img for each caption)
            scores[i_example] = score_table[i_example,i_example]

            # Calculate the rank of the correct image
            ordered_score_indices = np.argsort(score_table[i_example])[::-1]
            ranks[i_example] = np.where(ordered_score_indices == i_example)[0][0]

            # Calculate scores relative to the other top k closest
            if ranks[i_example] >= 10:
                other_top10 = ordered_score_indices[0:10]
            else:
                other_top10 = ordered_score_indices[np.delete(np.arange(11), ranks[i_example])]

            other_top5 = other_top10[0:5]

            relative_scores_at5[i_example] = scores[i_example] / np.mean(score_table[i_example][other_top5])
            relative_scores_at10[i_example] = scores[i_example] / np.mean(score_table[i_example][other_top10])

        # Compute metrics
        r1 = 100.0 * float(len(np.where(ranks < 1)[0])) / float(len(ranks))
        r5 = 100.0 * float(len(np.where(ranks < 5)[0])) / float(len(ranks))
        r10 = 100.0 * float(len(np.where(ranks < 10)[0])) / float(len(ranks))
        medr = np.floor(np.median(ranks)) + 1
        mean_rank = np.mean(ranks)

        # Find best and worst image + caption pairs
        best_scores = np.argsort(scores)

        return (r1, r5, r10, medr, mean_rank,
                np.mean(scores), np.mean(relative_scores_at5),
                np.mean(relative_scores_at10), best_scores)

    def t2i(self, images, captions):
        """
        Images: (5N, imgdim) matrix of images
        Captions: (5N, sentdim) matrix of captions
        """
        # Project images and captions
        img_embed, sent_embed = [], []
        for i in range(0, len(images), self.batch_size):
            img_embed.append(self.model.proj_image(
                Variable(images[i:i + self.batch_size], volatile=True)))
            sent_embed.append(self.model.proj_sentence(
                Variable(captions[i:i + self.batch_size], volatile=True)))
        img_embed = torch.cat(img_embed, 0).data
        sent_embed = torch.cat(sent_embed, 0).data

        npts = int(img_embed.size(0) / 5)
        idxs = torch.cuda.LongTensor(range(0, len(img_embed), 5))
        ims = img_embed.index_select(0, idxs)

        ranks = np.zeros(5 * npts)
        for index in range(npts):

            # Get query captions
            queries = sent_embed[5*index: 5*index + 5]

            # Compute scores
            scores = torch.mm(queries, ims.transpose(0, 1)).cpu().numpy()
            inds = np.zeros(scores.shape)
            for i in range(len(inds)):
                inds[i] = np.argsort(scores[i])[::-1]
                ranks[5 * index + i] = np.where(inds[i] == index)[0][0]

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        return (r1, r5, r10, medr)

    def i2t(self, images, captions):
        """
        Images: (5N, imgdim) matrix of images
        Captions: (5N, sentdim) matrix of captions
        """
        # Project images and captions
        img_embed, sent_embed = [], []
        for i in range(0, len(images), self.batch_size):
            img_embed.append(self.model.proj_image(
                Variable(images[i:i + self.batch_size], volatile=True)))
            sent_embed.append(self.model.proj_sentence(
                Variable(captions[i:i + self.batch_size], volatile=True)))
        img_embed = torch.cat(img_embed, 0).data
        sent_embed = torch.cat(sent_embed, 0).data

        npts = int(img_embed.size(0) / 5)
        index_list = []

        ranks = np.zeros(npts)
        for index in range(npts):

            # Get query image
            query_img = img_embed[5 * index]

            # Compute scores
            scores = torch.mm(query_img.view(1, -1),
                              sent_embed.transpose(0, 1)).view(-1)
            scores = scores.cpu().numpy()
            inds = np.argsort(scores)[::-1]
            index_list.append(inds[0])

            # Score
            rank = 1e20
            for i in range(5*index, 5*index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        return (r1, r5, r10, medr)
