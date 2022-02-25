# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file is originally from:
# https://github.com/ruotianluo/ImageCaptioning.pytorch
# It contains changes relating to the paper 'Generating Diverse and Meaningful
# Captions: Unsupervised Specificity Optimization for Image Captioning (Lindh
# et al., 2018)'. For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None


import torch
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

from operator import itemgetter
import time
import os
from six.moves import cPickle

from neuraltalk2_pytorch import models
from neuraltalk2_pytorch.dataloader import *
from neuraltalk2_pytorch.misc import utils

from SentEval.examples import infersent

from evaluate_model.eval_stats import store_generated_captions, calculate_distinct, calculate_novelty, calculate_vocabulary_usage

from .coco_caption2.pycocotools.coco import COCO
from .coco_caption2.pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

_USE_CUDA = True


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(opt):
    # Load the similarity bins
    print("Loading similarity bins...")
    least_similar, most_similar = _load_similarity_bins('val')
    print("...finished.")

    opt.use_att = utils.if_use_att(opt.caption_model)
    use_contrastive = opt.loss_function == 'pairwise_cosine' or opt.loss_function == 'pairwise_similarity'
    loader = DataLoader(opt, contrastive=use_contrastive)
    loader.seq_per_img = 1
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    starting_epoch = epoch

    accumulated_loss = 0

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    else:
        best_val_score = None

    model = models.setup(opt)
    if _USE_CUDA:
        model.cuda()

    # Load the SentEval model, the vocab needs to be word->index instead of index->word
    image_retriever = infersent.load_senteval({v: k for k, v in loader.get_vocab().items()},
                                              opt.batch_size, opt.loss_function, _USE_CUDA,
                                              pretrained_senteval_path=opt.senteval_model)

    # Assure in training mode
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        # If there's no optimizer file, skip this step (happens when first loading the pretrained model to start specificity-optimization)
        try:
            optimizer_state_dict = torch.load(os.path.join(opt.start_from, 'optimizer.pth'))
            # Workaround due to PyTorch issue 2647 https://github.com/pytorch/pytorch/issues/2647
            if len(optimizer_state_dict['state']) > 0:
                optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
            else:
                print("Optimizer state dict has empty state, using new optimizer.")
        except IOError:
            print("Optimizer file could not be loaded, using new optimizer. (This happens when you first start the specificity-optimization on a pretrained model.)")

    # Used for masking the END token gradients
    endtoken_index = torch.LongTensor([0]).cuda()

    while True:
        #start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train', seq_per_img=1)
        #print('Read data:', time.time() - start)

        if _USE_CUDA:
            torch.cuda.synchronize()
        start = time.time()

        # Stack the img_ids in the same way as the features
        stacked_ids = []
        if(use_contrastive):
            c_stacked_ids = []
        for i_infos in range( len(data['infos']) ):
            stacked_ids.append(data['infos'][i_infos]['ix'])
            if(use_contrastive):
                c_stacked_ids.append(data['c_infos'][i_infos]['ix'])

        stacked_ids = np.stack(stacked_ids)
        if(use_contrastive):
            c_stacked_ids = np.stack(c_stacked_ids)

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        if _USE_CUDA:
            tmp = [Variable(torch.from_numpy(_), requires_grad=True).cuda() for _ in tmp]
        else:
            tmp = [Variable(torch.from_numpy(_), requires_grad=True) for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        # Reset gradient calculations from the previous iteration
        optimizer.zero_grad()

        # Sample a batch of captions from the model, based on the image features
        sampled_seq_and_probs = model.sample(fc_feats, att_feats, filter_zerolength=True)
        valid_sequence_indices = sampled_seq_and_probs[3]

        # Let the IR system calculate the IR loss and the gradients wrt the input sequence
        if use_contrastive:
            train_loss, ir_gradients = image_retriever.caption_loss(stacked_ids[valid_sequence_indices],
                                                    sampled_seq_and_probs[0], sampled_seq_and_probs[2], split='train',
                                                    contrastive_imgids=c_stacked_ids[valid_sequence_indices])
        else:
            train_loss, ir_gradients = image_retriever.caption_loss(stacked_ids[valid_sequence_indices],
                                                    sampled_seq_and_probs[0], sampled_seq_and_probs[2], split='train')
        accumulated_loss += train_loss
        ir_gradients = ir_gradients.float()

        # Continue the backpropagation through the CG system from the probabilities
        # with the gradient from the IR system wrt the sampled sequence
        sampled_seq_and_probs[1].backward(ir_gradients)

        # Apply the backpropated gradients to the model parameters
        optimizer.step()

        torch.cuda.synchronize()
        end = time.time()

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1

        # Write the training loss summary
        if iteration % opt.losses_log_every == 0:
            average_loss = accumulated_loss.cpu().numpy()[0]/opt.losses_log_every
            accumulated_loss = 0
            np_train_loss = train_loss.cpu().numpy()[0]

            print("iter {} (starting_epoch {}, epoch {}), average_loss = {:.3f}, current train_loss = {:.3f}, time/batch = {:.3f}" \
                  .format(iteration, starting_epoch, epoch, average_loss, np_train_loss, end - start))

            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', np_train_loss, iteration)
                add_summary_value(tf_summary_writer, 'average_loss', average_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.learning_rate, iteration)
                tf_summary_writer.flush()

        # make evaluation on validation set, and save model
        if iteration % opt.save_checkpoint_every == 0:
            # Calculate the IR validation loss and other statistics
            (validation_loss, r1, r5, r10, median_rank, mean_rank,
             avg_score, avg_rel_score_at5, avg_rel_score_at10,
             distinct_captions, novel_captions, vocab_usage,
             _, _,
             least_r1, least_r5, least_r10, least_median_rank, least_mean_rank, least_distinct_captions,
             least_novel_captions, least_vocab_usage,
             most_r1, most_r5, most_r10, most_median_rank, most_mean_rank, most_distinct_captions,
             most_novel_captions, most_vocab_usage, lang_stats, least_lang_stats, most_lang_stats,
             avg_lengths, avg_duplicates, avg_lengths_least, avg_duplicates_least, avg_lengths_most, avg_duplicates_most
            ) = test(model=model, loader=loader, image_retriever=image_retriever, model_id=opt.id,
                                      epoch=str(epoch), iteration=str(iteration), split='val',
                                      use_contrastive=use_contrastive,
                                      least_similar=least_similar, most_similar=most_similar,
                                      language_eval=False)
            np_validation_loss = validation_loss.cpu().numpy()[0]

            print("Logging stats...")

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', np_validation_loss, iteration)
                add_summary_value(tf_summary_writer, 'distinct captions', distinct_captions, iteration)
                add_summary_value(tf_summary_writer, 'novel captions', novel_captions, iteration)
                add_summary_value(tf_summary_writer, 'r1', r1, iteration)
                add_summary_value(tf_summary_writer, 'r5', r5, iteration)
                add_summary_value(tf_summary_writer, 'r10', r10, iteration)
                add_summary_value(tf_summary_writer, 'median rank', median_rank, iteration)
                add_summary_value(tf_summary_writer, 'mean_rank', mean_rank, iteration)
                add_summary_value(tf_summary_writer, 'avg score', avg_score, iteration)
                add_summary_value(tf_summary_writer, 'avg relative score at 5', avg_rel_score_at5, iteration)
                add_summary_value(tf_summary_writer, 'avg relative score at 10', avg_rel_score_at10, iteration)
                add_summary_value(tf_summary_writer, 'avg_lengths', avg_lengths, iteration)
                add_summary_value(tf_summary_writer, 'avg_duplicates', avg_duplicates, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        add_summary_value(tf_summary_writer, k, v, iteration)

                tf_summary_writer.flush()

                add_summary_value(tf_summary_writer, 'least_distinct captions', least_distinct_captions, iteration)
                add_summary_value(tf_summary_writer, 'least_novel captions', least_novel_captions, iteration)
                add_summary_value(tf_summary_writer, 'least_r1', least_r1, iteration)
                add_summary_value(tf_summary_writer, 'least_r5', least_r5, iteration)
                add_summary_value(tf_summary_writer, 'least_r10', least_r10, iteration)
                add_summary_value(tf_summary_writer, 'least_median rank', least_median_rank, iteration)
                add_summary_value(tf_summary_writer, 'least_mean_rank', least_mean_rank, iteration)
                add_summary_value(tf_summary_writer, 'avg_lengths_least', avg_lengths_least, iteration)
                add_summary_value(tf_summary_writer, 'avg_duplicates_least', avg_duplicates_least, iteration)
                if least_lang_stats is not None:
                    for k,v in least_lang_stats.items():
                        add_summary_value(tf_summary_writer, 'least_' + k, v, iteration)

                tf_summary_writer.flush()

                add_summary_value(tf_summary_writer, 'most_distinct captions', most_distinct_captions, iteration)
                add_summary_value(tf_summary_writer, 'most_novel captions', most_novel_captions, iteration)
                add_summary_value(tf_summary_writer, 'most_r1', most_r1, iteration)
                add_summary_value(tf_summary_writer, 'most_r5', most_r5, iteration)
                add_summary_value(tf_summary_writer, 'most_r10', most_r10, iteration)
                add_summary_value(tf_summary_writer, 'most_median rank', most_median_rank, iteration)
                add_summary_value(tf_summary_writer, 'most_mean_rank', most_mean_rank, iteration)
                add_summary_value(tf_summary_writer, 'avg_lengths_most', avg_lengths_most, iteration)
                add_summary_value(tf_summary_writer, 'avg_duplicates_most', avg_duplicates_most, iteration)
                if most_lang_stats is not None:
                    for k, v in most_lang_stats.items():
                        add_summary_value(tf_summary_writer, 'most_' + k, v, iteration)

                tf_summary_writer.flush()

            print("validation_loss: ", np_validation_loss)
            print("distinct_captions: ", distinct_captions)
            print("novel_captions: ", novel_captions)
            if lang_stats is not None:
                print("lang_stats: ", lang_stats)
            print("r1: ", r1)
            print("r5: ", r5)
            print("r10: ", r10)
            print("median_rank: ", median_rank)
            print("mean_rank: ", mean_rank)
            print("avg_score: ", avg_score)
            print("avg_rel_score_at5: ", avg_rel_score_at5)
            print("avg_rel_score_at10: ", avg_rel_score_at10)
            # Save model if is improving on validation result
            if opt.best_model_condition == 'median_rank':
                current_score = -median_rank
            elif opt.best_model_condition == 'mean_rank':
                    current_score = -mean_rank
            elif opt.best_model_condition == 'rel_at_10':
                current_score = avg_rel_score_at10
            elif opt.best_model_condition == 'rel_at_5':
                current_score = avg_rel_score_at5
            elif opt.best_model_condition == 'r1':
                current_score = r1
            elif opt.best_model_condition == 'r5':
                current_score = r5
            elif opt.best_model_condition == 'r10':
                current_score = r10
            elif opt.best_model_condition == 'distinct_captions':
                current_score = distinct_captions
            elif opt.best_model_condition == 'cider':
                current_score = lang_stats['CIDEr']
            elif opt.best_model_condition == 'spice':
                current_score = lang_stats['SPICE']
            elif opt.best_model_condition == 'validation_loss':
                current_score = -np_validation_loss

            best_flag = False

            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True

            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()

            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                cPickle.dump(infos, f)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if (epoch-starting_epoch) >= opt.max_epochs and opt.max_epochs != -1:
            break


def test(language_eval=False, opt=None, model=None, loader=None, image_retriever=None, model_id=None, epoch=None,
         iteration=None, split=None, use_contrastive=None, least_similar=None, most_similar=None):
    # If no model is supplied, load the model from opt
    if model is None:
        if opt.split is None:
            print("Split was not defined, defaulting to test split.")
            opt.split = 'test'

        # Load the similarity bins
        print("Loading similarity bins...")
        least_similar, most_similar = _load_similarity_bins(opt.split)
        print("...finished.")

        opt.use_att = utils.if_use_att(opt.caption_model)
        use_contrastive = opt.loss_function == 'pairwise_cosine' or opt.loss_function == 'pairwise_similarity'
        loader = DataLoader(opt, use_contrastive)
        loader.seq_per_img = 1
        opt.vocab_size = loader.vocab_size
        opt.seq_length = loader.seq_length

        postfix = ''
        if opt.load_best_model == 1:
            postfix = '-best'
        # open old infos and check if models are compatible
        assert opt.start_from is not None, "start_from must be specified during test"
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+postfix+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        loader.iterators = infos.get('iterators', loader.iterators)
        loader.split_ix = infos.get('split_ix', loader.split_ix)

        model_id = opt.id
        epoch = str(infos.get('epoch', '0'))
        iteration = str(infos.get('iter', '0'))
        split = opt.split

        model = models.setup(opt)
        if _USE_CUDA:
            model.cuda()

        # Load the SentEval model, the vocab needs to be word->index instead of index->word
        image_retriever = infersent.load_senteval({v: k for k, v in loader.get_vocab().items()},
                                                  opt.batch_size, opt.loss_function, _USE_CUDA,
                                                  pretrained_senteval_path=opt.senteval_model)

    # Set to evaluation mode
    model.eval()
    image_retriever.params.infersent.eval()
    image_retriever.enable_learning(False)

    # Reset the loader to the start of the test set
    loader.reset_iterator(split)

    ir_split = split
    if ir_split == 'val':
        ir_split = 'valid'

    all_imgids = []
    all_captions = []
    all_lengths = []
    all_num_duplicate_words = []
    predictions = []

    all_imgids_least = []
    all_captions_least = []
    all_lengths_least = []
    all_num_duplicate_words_least = []
    predictions_least = []

    all_imgids_most = []
    all_captions_most = []
    all_lengths_most = []
    all_num_duplicate_words_most = []
    predictions_most = []

    test_loss = 0
    num_batches = 0
    is_final_batch = False

    # Run until the batches have gone through all the data and wrapped from start
    while True:
        # Get the next batch
        data = loader.get_batch(split, seq_per_img=1)

        # if we wrapped around the split then this is the final batch
        if data['bounds']['wrapped']:
            is_final_batch = True

        # Stack the img_ids in the same way as the features
        stacked_ids = []
        if use_contrastive:
            c_stacked_ids = []
        for i_infos in range( len(data['infos']) ):
            stacked_ids.append(data['infos'][i_infos]['ix'])
            if use_contrastive:
                c_stacked_ids.append(data['c_infos'][i_infos]['ix'])

        # Find which of these ids are part of the least/most similar bins
        indices_least = [i_id for i_id in range(len(stacked_ids)) if stacked_ids[i_id] in least_similar]
        least_in_batch = len(indices_least) > 0
        indices_most = [i_id for i_id in range(len(stacked_ids)) if stacked_ids[i_id] in most_similar]
        most_in_batch = len(indices_most) > 0

        # Keep the image ids for the evaluation
        all_imgids.extend(stacked_ids)
        if least_in_batch:
            all_imgids_least.extend([value for value in stacked_ids if value in least_similar])
        if most_in_batch:
            all_imgids_most.extend([value for value in stacked_ids if value in most_similar])

        # Arrange image ids as batch_size x 1
        stacked_ids = np.stack(stacked_ids)
        if use_contrastive:
            c_stacked_ids = np.stack(c_stacked_ids)

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        if _USE_CUDA:
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        else:
            tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        # Sample a batch of captions from the model, based on the image features
        sequences, _, seq_lengths, valid_sequence_indices = model.sample(fc_feats, att_feats, filter_zerolength=False)

        if use_contrastive:
            test_loss += image_retriever.caption_loss(stacked_ids[valid_sequence_indices], sequences, seq_lengths,
                                                      split=ir_split, loss_only=True,
                                                      contrastive_imgids=c_stacked_ids[valid_sequence_indices])
        else:
            test_loss += image_retriever.caption_loss(stacked_ids[valid_sequence_indices], sequences, seq_lengths,
                                                      split=ir_split, loss_only=True)
        num_batches += 1

        # Clean the captions from immediate duplicate words
        sequences, seq_lengths, num_duplicate_words = _clean_sequences(sequences)

        # Remember the captions for stats calculation
        all_captions.append(sequences)
        all_lengths.extend(seq_lengths)
        all_num_duplicate_words.extend(num_duplicate_words)

        if len(indices_least) > 0:
            if len(indices_least) == 1:
                # The itemgetter doesn't return a list when there's only 1 item
                current_lengths = [seq_lengths[indices_least[0]]]
                current_num_duplicates = [num_duplicate_words[indices_least[0]]]
            else:
                current_lengths = itemgetter(*indices_least)(seq_lengths)
                current_num_duplicates = itemgetter(*indices_least)(num_duplicate_words)
            all_lengths_least.extend(current_lengths)
            all_num_duplicate_words_least.extend(current_num_duplicates)
            all_captions_least.append(sequences[np.asarray(range(max(current_lengths))), :][:, np.asarray(indices_least)])

        if len(indices_most) > 0:
            if len(indices_most) == 1:
                # The itemgetter doesn't return a list when there's only 1 item
                current_lengths = [seq_lengths[indices_most[0]]]
                current_num_duplicates = [num_duplicate_words[indices_most[0]]]
            else:
                current_lengths = itemgetter(*indices_most)(seq_lengths)
                current_num_duplicates = itemgetter(*indices_most)(num_duplicate_words)
            all_lengths_most.extend(current_lengths)
            all_num_duplicate_words_most.extend(current_num_duplicates)
            all_captions_most.append(sequences[np.asarray(range(max(current_lengths))), :][:, np.asarray(indices_most)])

        # Store info about decoded captions for language evaluation
        decoded_captions = utils.decode_sequence(loader.get_vocab(), sequences.transpose(0, 1).cpu().numpy())
        predictions.extend([ {'image_id': data['infos'][i_cap]['id'], 'caption': decoded_captions[i_cap]} for i_cap in range( len(decoded_captions) ) ])

        if least_in_batch:
            decoded_captions = utils.decode_sequence(loader.get_vocab(), sequences[:,indices_least].transpose(0, 1).cpu().numpy())
            predictions_least.extend([ {'image_id': data['infos'][i_cap]['id'], 'caption': decoded_captions[i_cap]} for i_cap in range( len(decoded_captions) ) ])

        if most_in_batch:
            decoded_captions = utils.decode_sequence(loader.get_vocab(), sequences[:,indices_most].transpose(0, 1).cpu().numpy())
            predictions_most.extend([ {'image_id': data['infos'][i_cap]['id'], 'caption': decoded_captions[i_cap]} for i_cap in range( len(decoded_captions) ) ])

        if is_final_batch:
            break

    # The final batch might be smaller than the normal batch size
    final_batch_size = sequences.size(1)

    # Print current predictions
    print("PREDICTIONS", predictions)

    if not os.path.isdir('test_results'):
        os.mkdir('test_results')
    caption_file = os.path.join('test_results/', model_id + '_' + split + '.json')
    cache_path = os.path.join('test_results/', model_id + '_' + split + '_tmp.json')
    # This dump is needed for language evaluation and diversity stats
    json.dump(predictions, open(cache_path, 'w'))

    lang_stats = None
    # Calculate language stats
    if language_eval:
        annFile = os.path.join(os.getcwd(), 'neuraltalk2_pytorch/coco_caption/annotations/captions_val2014.json')

        coco = COCO(annFile)

        # Load our predictions (requires a few steps)
        cocoRes = coco.loadRes(cache_path)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        # Evaluate our predictions
        cocoEval.evaluate()

        # Prepare the output dictionary
        lang_stats = {}
        for metric, score in cocoEval.eval.items():
            lang_stats[metric] = score

        imgToEval = cocoEval.imgToEval
        for p in predictions:
            image_id, caption = p['image_id'], p['caption']
            imgToEval[image_id]['caption'] = caption
            imgToEval[image_id]['SPICE'] = imgToEval[image_id]['SPICE']['All']['f']

        # Export the results
        with open(caption_file, 'w') as outfile:
            json.dump({'overall': lang_stats, 'imgToEval': imgToEval}, outfile)

    # Calculate specificity stats
    all_imgids = np.stack(all_imgids)
    all_lengths = np.stack(all_lengths)
    batch_size = all_captions[0].size(1)
    num_examples = batch_size*len(all_captions)
    if final_batch_size < batch_size:
        num_examples -= (batch_size - final_batch_size)
    all_captions_stacked = torch.zeros(np.max(all_lengths), num_examples).long()
    for i_caption in range(len(all_captions)):
        column_end = (i_caption+1)*batch_size
        if column_end > num_examples:
            column_end = num_examples
        all_captions_stacked[0:all_captions[i_caption].size(0), i_caption*batch_size:column_end] = all_captions[i_caption].long()
    all_captions = None
    if _USE_CUDA:
        all_captions_stacked = all_captions_stacked.cuda()

    (r1, r5, r10, median_rank, mean_rank,
     avg_score, avg_rel_score_at5, avg_rel_score_at10,
     _) = image_retriever.t2i_stats(all_imgids, all_captions_stacked, all_lengths, ir_split)

    # Calculate and print diversity stats
    if opt is None:
        evaluation_table_name = 'validation_' + model_id + '_' + split
        evaluation_table_name_least = 'validation_' + model_id + '_' + split + '_least'
        evaluation_table_name_most = 'validation_' + model_id + '_' + split + '_most'
    else:
        evaluation_table_name = 'testrun_' + model_id + '_' + split
        evaluation_table_name_least = 'testrun_' + model_id + '_' + split + '_least'
        evaluation_table_name_most = 'testrun_' + model_id + '_' + split + '_most'

    current_split = model_id + '_epoch' + epoch + '_iter' + iteration
    store_generated_captions(cache_path, evaluation_table_name, current_split)
    distinct_captions = calculate_distinct(evaluation_table_name, [current_split], False)
    novel_captions = calculate_novelty(evaluation_table_name, current_split, False)
    vocab_usage = calculate_vocabulary_usage(evaluation_table_name, current_split, False)

    # Record sequence stats
    all_num_duplicate_words = np.stack(all_num_duplicate_words)
    avg_duplicates = np.mean(all_num_duplicate_words)
    avg_lengths = np.mean(all_lengths)


    ### Least similar
    caption_file = os.path.join('test_results/', model_id + '_' + split + '_least.json')
    cache_path = os.path.join('test_results/', model_id + '_' + split + '_least_tmp.json')
    json.dump(predictions_least, open(cache_path, 'w'))

    least_lang_stats = None
    if language_eval:
        coco = COCO(annFile)

        # Load our predictions (requires a few steps)
        cocoRes = coco.loadRes(cache_path)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        # Evaluate our predictions
        cocoEval.evaluate()

        # Prepare the output dictionary
        least_lang_stats = {}
        for metric, score in cocoEval.eval.items():
            least_lang_stats[metric] = score

        imgToEval = cocoEval.imgToEval
        for p in predictions_least:
            image_id, caption = p['image_id'], p['caption']
            imgToEval[image_id]['caption'] = caption
            imgToEval[image_id]['SPICE'] = imgToEval[image_id]['SPICE']['All']['f']
        # Export the results
        with open(caption_file, 'w') as outfile:
            json.dump({'overall': least_lang_stats, 'imgToEval': imgToEval}, outfile)

    all_imgids_least = np.stack(all_imgids_least)
    all_lengths_least = np.stack(all_lengths_least)

    all_captions_stacked_least = torch.zeros(np.max(all_lengths_least), len(all_imgids_least)).long()
    i_least = 0
    for current_captions in all_captions_least:
        all_captions_stacked_least[0:current_captions.size(0), i_least:i_least+current_captions.size(1)] = current_captions
        i_least += current_captions.size(1)
    all_captions_least = None
    if _USE_CUDA:
        all_captions_stacked_least = all_captions_stacked_least.cuda()

    (least_r1, least_r5, least_r10, least_median_rank, least_mean_rank,
     _, _, _,
     _) = image_retriever.t2i_stats(all_imgids_least, all_captions_stacked_least, all_lengths_least, ir_split)

    store_generated_captions(cache_path, evaluation_table_name_least, current_split)
    least_distinct_captions = calculate_distinct(evaluation_table_name_least, [current_split], False)
    least_novel_captions = calculate_novelty(evaluation_table_name_least, current_split, False)
    least_vocab_usage = calculate_vocabulary_usage(evaluation_table_name_least, current_split, False)

    # Record sequence stats
    all_num_duplicate_words_least = np.stack(all_num_duplicate_words_least)
    avg_duplicates_least = np.mean(all_num_duplicate_words_least)
    avg_lengths_least = np.mean(all_lengths_least)


    ### Most similar
    caption_file = os.path.join('test_results/', model_id + '_' + split + '_most.json')
    cache_path = os.path.join('test_results/', model_id + '_' + split + '_most_tmp.json')
    json.dump(predictions_most, open(cache_path, 'w'))

    most_lang_stats = None
    if language_eval:
        coco = COCO(annFile)

        # Load our predictions (requires a few steps)
        cocoRes = coco.loadRes(cache_path)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        # Evaluate our predictions
        cocoEval.evaluate()

        # Prepare the output dictionary
        most_lang_stats = {}
        for metric, score in cocoEval.eval.items():
            most_lang_stats[metric] = score

        imgToEval = cocoEval.imgToEval
        for p in predictions_most:
            image_id, caption = p['image_id'], p['caption']
            imgToEval[image_id]['caption'] = caption
            imgToEval[image_id]['SPICE'] = imgToEval[image_id]['SPICE']['All']['f']
        # Export the results
        with open(caption_file, 'w') as outfile:
            json.dump({'overall': most_lang_stats, 'imgToEval': imgToEval}, outfile)

    all_imgids_most = np.stack(all_imgids_most)
    all_lengths_most = np.stack(all_lengths_most)
    all_captions_stacked_most = torch.zeros(np.max(all_lengths_most), len(all_imgids_most)).long()

    i_most = 0
    for current_captions in all_captions_most:
        all_captions_stacked_most[0:current_captions.size(0), i_most:i_most+current_captions.size(1)] = current_captions
        i_most += current_captions.size(1)
    all_captions_most = None
    if _USE_CUDA:
        all_captions_stacked_most = all_captions_stacked_most.cuda()

    (most_r1, most_r5, most_r10, most_median_rank, most_mean_rank,
     _, _, _,
     _) = image_retriever.t2i_stats(all_imgids_most, all_captions_stacked_most, all_lengths_most, ir_split)

    store_generated_captions(cache_path, evaluation_table_name_most, current_split)
    most_distinct_captions = calculate_distinct(evaluation_table_name_most, [current_split], False)
    most_novel_captions = calculate_novelty(evaluation_table_name_most, current_split, False)
    most_vocab_usage = calculate_vocabulary_usage(evaluation_table_name_most, current_split, False)

    # Record sequence stats
    all_num_duplicate_words_most = np.stack(all_num_duplicate_words_most)
    avg_duplicates_most = np.mean(all_num_duplicate_words_most)
    avg_lengths_most = np.mean(all_lengths_most)


    if opt is None:
        # Switch back to training mode
        model.train()
        image_retriever.params.infersent.train()
        image_retriever.enable_learning(True)

    return (test_loss / num_batches, r1, r5, r10, median_rank, mean_rank,
            avg_score, avg_rel_score_at5, avg_rel_score_at10,
            distinct_captions, novel_captions, vocab_usage,
            evaluation_table_name, current_split,
            least_r1, least_r5, least_r10, least_median_rank, least_mean_rank, least_distinct_captions,
            least_novel_captions, least_vocab_usage,
            most_r1, most_r5, most_r10, most_median_rank, most_mean_rank, most_distinct_captions,
            most_novel_captions, most_vocab_usage, lang_stats, least_lang_stats, most_lang_stats,
            avg_lengths, avg_duplicates, avg_lengths_least, avg_duplicates_least, avg_lengths_most, avg_duplicates_most)


# Remove immediate duplicate tokens in sequences
def _clean_sequences(sequences):
    cleaned_sequences = sequences.new(sequences.size()).zero_()
    max_length = sequences.size(0)
    sequence_lengths = []
    all_num_duplicate_words = []

    # Copy over the non-duplicate sequence parts
    for i_seq in range(sequences.size(1)):
        i_original_start = 0
        i_cleaned_start = 0
        i_cleaned_end = 0
        num_duplicate_words = 0

        # Loop through everything before the end token (0) for this sequence
        while (i_original_start < max_length) and (sequences[i_original_start, i_seq] != 0):
            # Update the current token to our copy current position
            current_token = sequences[i_original_start, i_seq]

            # Set the end token to be the next token
            i_original_end = i_original_start + 1

            # Increment the end of this copy sequence as long as there are no duplicate tokens
            while (i_original_end < max_length and sequences[i_original_end, i_seq] != 0 and
                   sequences[i_original_end, i_seq] != current_token):
                current_token = sequences[i_original_end, i_seq]
                i_original_end += 1

            # Calculate the end row index in the target matrix
            i_cleaned_end = i_cleaned_start + (i_original_end - i_original_start)

            # Copy over the current sequence
            cleaned_sequences[i_cleaned_start:i_cleaned_end, i_seq] = sequences[i_original_start:i_original_end, i_seq]

            # Set the next start of indices to where the current sequence piece ended
            i_cleaned_start = i_cleaned_end
            i_original_start = i_original_end

            # Increment the start index beyond the last duplicate token
            while (i_original_start < max_length and sequences[i_original_start, i_seq] != 0 and
                   sequences[i_original_start, i_seq] == current_token):
                i_original_start += 1
                num_duplicate_words += 1

        # Keep track of the new sequence lengths and number of removed duplicate words
        sequence_lengths.append(i_cleaned_end)
        all_num_duplicate_words.append(num_duplicate_words)

    # Return the actually used part of the cleaned sequence matrix, along with the new lengths
    max_length = max(sequence_lengths)
    return cleaned_sequences[0:max_length,:], sequence_lengths, all_num_duplicate_words


def _load_similarity_bins(split):
    data = np.load(os.path.join('data/similarity_stats_mean_20p_' + split + '.npz'))

    return data['least_similar_20p'], data['most_similar_20p']
