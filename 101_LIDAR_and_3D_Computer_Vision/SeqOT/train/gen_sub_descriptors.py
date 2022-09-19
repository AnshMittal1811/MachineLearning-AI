#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: generate sub-descriptors for the gem training and evaluation


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')
import torch
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm
import yaml
from modules.seqTransformerCat import featureExtracter
from tools.read_samples import read_one_need_from_seq
from tools.utils.utils import *


class testHandler():
    def __init__(self, seqlen=3, pretrained_weights=None,range_image_database_root=None,
                               range_image_query_root=None):
        super(testHandler, self).__init__()

        self.seq_len = seqlen
        self.amodel = featureExtracter(seqL=self.seq_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        print(self.amodel)
        self.parameters  = self.amodel.parameters()
        self.weights = pretrained_weights
        self.range_image_database_root = range_image_database_root
        self.range_image_query_root = range_image_query_root

    def eval(self):

        resume_filename = self.weights
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        self.amodel.load_state_dict(checkpoint['state_dict'])  # 加载状态字典

        interval = 1

        scan_paths_database = load_files(self.range_image_database_root)
        print("the number of reference scans ", len(scan_paths_database))
        des_list = np.zeros((int(len(scan_paths_database)//interval)+1, 256))
        for j in tqdm(np.arange(0, len(scan_paths_database), interval)):
            current_batch, read_complete_flag = read_one_need_from_seq(str(j).zfill(6), self.seq_len, range_image_root=self.range_image_database_root)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            des_list[int(j//interval), :] = current_batch_des[0, :].cpu().detach().numpy()
        des_list = des_list.astype('float32')
        np.save("des_list_database", des_list)


        scan_paths_query = load_files(self.range_image_query_root)
        print("the number of query scans ", len(scan_paths_query))
        des_list_query = np.zeros((int(len(scan_paths_query)//interval)+1, 256))
        for j in tqdm(np.arange(0, len(scan_paths_query), interval)):
            current_batch, read_complete_flag = read_one_need_from_seq(str(j).zfill(6), self.seq_len, range_image_root=self.range_image_query_root)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            des_list_query[int(j//interval), :] = current_batch_des[0, :].cpu().detach().numpy()
        des_list_query = des_list_query.astype('float32')
        np.save("des_list_query", des_list_query)

if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    seqlen = config["gen_sub_descriptors"]["seqlen"]
    pretrained_weights = config["gen_sub_descriptors"]["weights"]
    range_image_database_root = config["data_root"]["range_image_database_root"]
    range_image_query_root = config["data_root"]["range_image_query_root"]
    # ============================================================================
    test_handler = testHandler(seqlen=seqlen, pretrained_weights=pretrained_weights,
                               range_image_database_root=range_image_database_root,
                               range_image_query_root=range_image_query_root)
    test_handler.eval()