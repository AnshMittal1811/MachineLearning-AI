#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: read sampled range images or descriptors as single input or batch input


import torch
import numpy as np
import sys
sys.path.append('../tools/')
sys.path.append('../modules/')
np.set_printoptions(threshold=sys.maxsize)
from utils.utils import *


def read_one_need_from_seq(file_num, seq_len, poses=None, range_image_root=None):
    read_complete_flag = True
    depth_data_seq = torch.zeros((1, seq_len, 32, 900)).type(torch.FloatTensor).cuda()
    for i in np.arange(int(file_num)-(seq_len//2), int(file_num)-(seq_len//2)+seq_len):
        file_num_str = str(i).zfill(6)
        if not os.path.exists(range_image_root+file_num_str+".npy"):
            read_complete_flag = False
            depth_data_tmp = np.load(range_image_root+file_num+".npy")
            depth_data_tensor_tmp = torch.from_numpy(depth_data_tmp).type(torch.FloatTensor).cuda()
            depth_data_tensor_tmp = torch.unsqueeze(depth_data_tensor_tmp, dim=0)
            depth_data_tensor_tmp = torch.unsqueeze(depth_data_tensor_tmp, dim=0)
            for m in np.arange(int(file_num) - (seq_len // 2), int(file_num) - (seq_len // 2) + seq_len):
                depth_data_seq[:, int(m - int(file_num) + (seq_len // 2)), :, :] = depth_data_tensor_tmp
            return depth_data_seq, read_complete_flag
        depth_data = np.load(range_image_root+file_num_str+".npy")
        depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
        depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
        depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
        depth_data_seq[:,int(i-int(file_num)+(seq_len//2)),:,: ] = depth_data_tensor

    return depth_data_seq, read_complete_flag





def read_one_batch_pos_neg(f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2, range_image_root,
                           train_overlap, overlap_thresh,seq_len, poses=None):
    read_complete_flag = True
    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt] and (train_overlap[tt]> overlap_thresh or train_overlap[tt]<(overlap_thresh-0.0)):
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, seq_len, 32, 900))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> overlap_thresh:
                pos_num = pos_num + 1
                pos_flag = True
            elif train_overlap[j]< overlap_thresh - 0.0:
                neg_num = neg_num + 1
            else:
                continue

            depth_data_seq = torch.zeros((seq_len, 32, 900)).type(torch.FloatTensor).cuda()
            for i in np.arange(int(train_imgf2[j]) - (seq_len // 2),
                               int(train_imgf2[j]) - (seq_len // 2) + seq_len):  # length can be changed !!!!!!!
                file_num_str = str(i).zfill(6)
                if not os.path.exists(range_image_root + file_num_str + ".npy"):
                    read_complete_flag = False
                    return sample_batch, sample_truth, pos_num, neg_num, read_complete_flag
                depth_data = np.load(range_image_root + file_num_str + ".npy")
                depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
                depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
                depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
                depth_data_seq[int(i - int(train_imgf2[j]) + (seq_len // 2)), :, :] = depth_data_tensor

            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_data_seq
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_data_seq
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1

    return sample_batch, sample_truth, pos_num, neg_num, read_complete_flag



def read_one_need_descriptor_from_seq_ft(file_num, descriptors, seq_len, poses=None):
    read_complete_flag = True
    descriptors_seq = torch.zeros((1, seq_len, 256)).type(torch.FloatTensor).cuda()
    for i in np.arange(int(file_num)-(seq_len//2), int(file_num)-(seq_len//2)+seq_len):  # length can be changed !!!!!!!
        if i<0 or i>=descriptors.shape[0]:
            read_complete_flag = False
            for m in np.arange(int(file_num) - (seq_len // 2), int(file_num) - (seq_len // 2) + seq_len):
                descriptors_seq[0, int(m - int(file_num) + (seq_len // 2)), :] = torch.from_numpy(descriptors[int(file_num),:]).type(torch.FloatTensor).cuda()
            return descriptors_seq, read_complete_flag
        descriptor_tensor = torch.from_numpy(descriptors[i,:]).type(torch.FloatTensor).cuda()
        descriptors_seq[0,int(i-int(file_num)+(seq_len//2)),:] = descriptor_tensor

    return descriptors_seq, read_complete_flag



def read_one_batch_pos_neg_descriptors(f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, overlap_thresh, seq_len, descs):
    read_complete_flag = True
    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt] and (train_overlap[tt]> overlap_thresh or train_overlap[tt]<(overlap_thresh-0.0)):
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, seq_len, 256))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> overlap_thresh:
                pos_num = pos_num + 1
                pos_flag = True
            elif train_overlap[j]< overlap_thresh - 0.0:
                neg_num = neg_num + 1
            else:
                continue

            depth_data_seq, read_complete_flag = read_one_need_descriptor_from_seq_ft(train_imgf2[j], descs, seq_len)
            if not read_complete_flag:
                return sample_batch, sample_truth, pos_num, neg_num, read_complete_flag

            if pos_flag:
                sample_batch[pos_idx,:,:] = depth_data_seq
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :] = depth_data_seq
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1

    return sample_batch, sample_truth, pos_num, neg_num, read_complete_flag