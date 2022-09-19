#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: generate predictions for the final evaluation


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')
import torch
import yaml
from tqdm import tqdm
import faiss
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from modules.gem import GeM
from tools.read_samples import read_one_need_descriptor_from_seq_ft
from tools.utils.utils import *


class testHandler():
    def __init__(self, seqlen=20, pretrained_weights=None, descs_database=None, descs_query=None):
        super(testHandler, self).__init__()

        self.amodel = GeM()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.weights = pretrained_weights
        self.descs_database = descs_database
        self.descs_query = descs_query
        self.seqlen = seqlen

    def eval(self):

        resume_filename = self.weights
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        self.amodel.load_state_dict(checkpoint['state_dict'])

        #########################################################################################################################
        interval = 10

        des_list = np.zeros((int(self.descs_database.shape[0]//interval)+1, 256))

        for j in tqdm(np.arange(0, self.descs_database.shape[0], interval)):
            f1_index = str(j).zfill(6)
            current_batch,_ = read_one_need_descriptor_from_seq_ft(f1_index, self.descs_database, seq_len=self.seqlen)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            current_batch_des = current_batch_des.squeeze(1)
            des_list[int(j//interval), :] = current_batch_des[0, :].cpu().detach().numpy()
        des_list = des_list.astype('float32')

        row_list = []
        nlist = 1
        k = 22
        d = 256
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained
        index.train(des_list)
        assert index.is_trained
        index.add(des_list)

        for i in range(0, self.descs_query.shape[0], 5):
            i_index = str(i).zfill(6)
            current_batch,_ = read_one_need_descriptor_from_seq_ft(i_index, self.descs_query, seq_len=self.seqlen)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            current_batch_des = current_batch_des.squeeze(1)
            des_list_current = current_batch_des[0, :].cpu().detach().numpy()
            D, I = index.search(des_list_current.reshape(1, -1), k)

            for j in range(D.shape[1]):
                one_row = np.zeros((1,3))
                one_row[:, 0] = i
                one_row[:, 1] = I[:,j]*interval
                one_row[:, 2] = D[:,j]
                row_list.append(one_row)
                print("ref:"+str(i) + "---->" + "query:" + str(I[:, j]*interval ) + "  " + str(D[:, j]))

        row_list_arr = np.array(row_list)
        np.savez_compressed("./predicted_L2_dis", row_list_arr)


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    sub_descriptors_database = np.load(config["test_gem_prepare"]["sub_descriptors_database_file"])
    sub_descriptors_query = np.load(config["test_gem_prepare"]["sub_descriptors_query_file"])
    seqlen = config["test_gem_prepare"]["seqlen"]
    pretrained_weights = config["test_gem_prepare"]["weights"]
    # ============================================================================
    test_handler = testHandler(seqlen=seqlen, pretrained_weights=pretrained_weights,
                               descs_database=sub_descriptors_database,
                               descs_query=sub_descriptors_query)

    test_handler.eval()