# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper 'Generating Diverse
# and Meaningful Captions: Unsupervised Specificity Optimization for Image
# Captioning (Lindh et al., 2018)'
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import math
import sys
import os
sys.path.append(os.getcwd())

from neuraltalk2_pytorch import opts
from neuraltalk2_pytorch.dataloader import *

_SAVE_DIR = 'data'

def calculate_similarities(opt):
    opt.use_att = False  # We don't need to load the attention features for this
    opt.seq_per_img = 1

    # Create two dataloader to allow comparing each batch to the rest
    loader_other = DataLoader(opt, allow_shuffle=False)
    loader_main = DataLoader(opt, allow_shuffle=False)

    opt.seq_length = loader_other.seq_length

    num_examples = loader_main.get_split_size(opt.split)
    print("num_examples", num_examples)
    num_batches = int(math.ceil(float(num_examples) / float(opt.batch_size)))
    print("num_batches", num_batches)
    num_1p = int(math.ceil(float(num_examples) / 100.0))
    num_20p = int(math.ceil(20.0 * float(num_examples) / 100.0))

    if(opt.split != 'train'):
        # Mean top 1% similarities for each example
        top_1p_similarities = np.zeros([num_examples], dtype=np.float32)

        # Use these indices when calculating the mean of the top 1% for each row in the current batch
        row_indices = np.repeat(range(opt.batch_size), num_1p)

    # Use these indices to mask out the similarities within a batch between the same image
    diagonal_indices = torch.LongTensor(range(opt.batch_size)).cuda()

    # Store the top 1% image indices in order of the similarity scores for each image
    sorted_image_indices = np.zeros([num_examples, num_1p], dtype=np.int32)
    # For convenience, keep track of overall index order
    all_indices = np.zeros([num_examples], dtype=np.int32)
    # Keep track of the end index for copying the results
    last_primary_example = 0

    for i_primary_batch in range(num_batches):
        # Load the image features for the primary batch
        data_main = loader_main.get_batch(opt.split, seq_per_img=1)

        # Keep track of the end index for copying the results
        current_primary_batch_size = len(data_main['fc_feats'])
        last_primary_example += current_primary_batch_size

        # Store the similarities to this batch of images
        current_similarities = torch.FloatTensor(current_primary_batch_size, num_examples).cuda().zero_()

        features_main = torch.FloatTensor(data_main['fc_feats']).cuda()

        # Keep track of the end index for copying the results
        last_secondary_example = 0

        for i_secondary_batch in range(num_batches):
            data_secondary = loader_other.get_batch(opt.split, seq_per_img=1)

            # Keep track of the end index for copying the results
            last_secondary_example += len(data_secondary['fc_feats'])

            # Get the fc7 features for the images
            features_secondary = torch.FloatTensor(data_secondary['fc_feats']).cuda()

            # Calculate dot product similarities between all images in current batches
            sim = torch.mm(features_main, features_secondary.transpose(0,1))
            # Normalize the similarities
            scale_main = torch.sqrt(torch.pow(features_main, 2).sum(1, keepdim=True))
            scale_secondary = torch.sqrt(torch.pow(features_secondary, 2).sum(1, keepdim=True))
            sim = sim / torch.mm(scale_main, scale_secondary.transpose(0, 1))

            # Mask out the similarities between the same images
            if i_primary_batch == i_secondary_batch:
                # If the final batch has a different size, create a special diagonal for this one
                if current_primary_batch_size != opt.batch_size:
                    diagonal_indices = torch.LongTensor(range(current_primary_batch_size)).cuda()

                sim[diagonal_indices, diagonal_indices] = -10.0

            # Copy to the np array
            current_similarities[:, (i_secondary_batch * opt.batch_size):last_secondary_example] = sim

            # Keep track of the index order (which will be the same since we're not shuffling)
            if i_primary_batch == 0:
                all_indices[(i_secondary_batch * opt.batch_size):last_secondary_example] = np.array([current_info['ix'] for current_info in data_secondary['infos']], dtype=np.int32)

        # Sort by highest similarity and keep top 1 percent
        current_sims_cpu = current_similarities.cpu().numpy()
        sorted_indices = np.argsort(current_sims_cpu)[:,::-1][:,0:num_1p]

        # The first example is skipped since it should be the image itself
        sorted_image_indices[(i_primary_batch * opt.batch_size):last_primary_example, :] = all_indices[sorted_indices]

        # Calculate the mean similarity to this image's top 1% most similar
        if opt.split != 'train':
            if current_primary_batch_size != opt.batch_size:
                row_indices = row_indices[0:current_primary_batch_size*num_1p]
            top_1p_similarities[(i_primary_batch * opt.batch_size):last_primary_example] = np.mean(
                current_sims_cpu[row_indices, sorted_indices.ravel()].reshape(current_primary_batch_size,
                                                                                         num_1p), axis=1)

    # Save the contrastive image indices
    np.savez(os.path.join(_SAVE_DIR, 'similarity_stats_top_1p_' + opt.split),
                num_1p=np.asarray([num_1p]),
                sorted_image_indices=sorted_image_indices,
                all_indices=all_indices
            )

    # Find the 20% most and least similar images of this dataset
    if opt.split != 'train':
        reverse_sorted_mean_similarity_indices = np.argsort(top_1p_similarities)
        least_similar_20p = all_indices[reverse_sorted_mean_similarity_indices[0:num_20p]]
        most_similar_20p = all_indices[reverse_sorted_mean_similarity_indices[-num_20p:][::-1]]

        # Save the most/least similar indices
        np.savez(os.path.join(_SAVE_DIR, 'similarity_stats_mean_20p_' + opt.split),
                     num_20p=np.asarray([num_20p]),
                     least_similar_20p=least_similar_20p,
                     most_similar_20p=most_similar_20p
                )


# Print the filepaths to the some images and one of their top 1% most similar for manual verification
# (To get the top most similar, remove the randomness from dataloader.py)
def verify_results(options):
    options.use_att = False  # We don't need to load the attention features for this
    options.seq_per_img = 5

    loader = DataLoader(options, contrastive=True, allow_shuffle=False)

    options.seq_length = loader.seq_length
    batch = loader.get_batch(options.split)
    print(batch['infos'], batch['c_infos'])


if __name__ == '__main__':
    opt = opts.parse_opt()
    calculate_similarities(opt)
    #verify_results(opt)
