#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: architecture of SeqOT


import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../tools/')
from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F

class featureExtracter(nn.Module):
    def __init__(self, seqL=5):
        super(featureExtracter, self).__init__()

        self.seqL = seqL

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2,1), stride=(2,1), bias=False)
        self.conv1_add = nn.Conv2d(16, 16, kernel_size=(5,1), stride=(1,1), bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)

        self.relu = nn.ReLU(inplace=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        encoder_layer2 = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)

        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.convLast2 = nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1), bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.net_vlad = NetVLADLoupe(feature_size=512, max_samples=int(900*self.seqL), cluster_size=64,  # before 11.12 --- 64
                                     output_dim=256, gating=True, add_batch_norm=False,   # output_dim=512
                                     is_training=True)


    def forward(self, x_l):
        out_l_seq = None
        for i in range(self.seqL):

            one_x_l_from_seq = x_l[:, i:(i+1), :, :]

            out_l = self.relu(self.conv1(one_x_l_from_seq))
            out_l = self.relu(self.conv1_add(out_l))
            out_l = self.relu(self.conv2(out_l))
            out_l = self.relu(self.conv3(out_l))
            out_l = self.relu(self.conv4(out_l))
            out_l = self.relu(self.conv5(out_l))
            out_l = self.relu(self.conv6(out_l))
            out_l = self.relu(self.conv7(out_l))


            out_l_1 = out_l.permute(0,1,3,2)
            out_l_1 = self.relu(self.convLast1(out_l_1))
            out_l = out_l_1.squeeze(3)
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)
            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            if i==0:
                out_l_seq = out_l
            else:
                out_l_seq = torch.cat((out_l_seq, out_l), dim=-2)

        out_l_seq = out_l_seq.squeeze(3)
        out_l_seq = out_l_seq.permute(2, 0, 1)
        out_l_seq = self.transformer_encoder2(out_l_seq)
        out_l_seq = out_l_seq.permute(1, 2, 0)
        out_l_seq = out_l_seq.unsqueeze(3)
        out_l_seq = self.net_vlad(out_l_seq)
        out_l_seq = F.normalize(out_l_seq, dim=1)
        return out_l_seq



if __name__ == '__main__':
    amodel = featureExtracter(5)
    print(amodel)
