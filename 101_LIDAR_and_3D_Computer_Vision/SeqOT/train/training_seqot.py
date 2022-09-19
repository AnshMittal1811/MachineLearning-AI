#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: train SeqOT with the database of the NCLT dataset


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
import yaml
from tensorboardX import SummaryWriter
from modules.seqTransformerCat import featureExtracter
import tools.loss as PNV_loss
from tools.read_samples import read_one_need_from_seq
from tools.read_samples import read_one_batch_pos_neg
from tools.utils.utils import *


class trainHandler():
    def __init__(self, height=32, width=900, seqlen=3, lr=0.000005, resume=False, pretrained_weights=None,
                 train_set=None, poses_file=None, range_image_root=None):
        super(trainHandler, self).__init__()

        self.height = height
        self.width = width
        self.seq_len = seqlen
        self.learning_rate = lr
        self.resume = resume
        self.train_set = train_set
        self.poses_file = poses_file
        self.weights = pretrained_weights
        self.range_image_root = range_image_root

        self.amodel = featureExtracter(seqL=self.seq_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.parameters  = self.amodel.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        self.traindata_npzfiles = train_set
        self.train_set_imgf1_imgf2_overlap = np.load(self.train_set)
        self.poses = np.load(self.poses_file)
        self.overlap_thresh = 0.3

    def train(self):
        epochs = 100

        if self.resume:
            resume_filename = self.weights
            print("Resuming From ", resume_filename)
            checkpoint = torch.load(resume_filename)
            starting_epoch = checkpoint['epoch']
            self.amodel.load_state_dict(checkpoint['state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Training From Scratch ..." )
            starting_epoch = 0

        writer1 = SummaryWriter(comment="LR_xxx")

        for i in range(starting_epoch+1, epochs):

            # shuffle
            # self.train_set_imgf1_imgf2_overlap = np.random.permutation(self.train_set_imgf1_imgf2_overlap)

            self.train_imgf1 = self.train_set_imgf1_imgf2_overlap[:, 0]
            self.train_imgf2 = self.train_set_imgf1_imgf2_overlap[:, 1]
            self.train_dir1 = np.zeros((len(self.train_imgf1),))
            self.train_dir2 = np.zeros((len(self.train_imgf2),))
            self.train_overlap = self.train_set_imgf1_imgf2_overlap[:, 2].astype(float)

            print("=======================================================================\n\n\n")
            print("total pairs: ", len(self.train_imgf1))
            print("\n\n\n=======================================================================")

            loss_each_epoch = 0
            used_num = 0

            used_list_f1 = []
            used_list_dir1 = []

            for j in range(len(self.train_imgf1)):
                f1_index = self.train_imgf1[j]
                dir1_index = self.train_dir1[j]
                continue_flag = False
                for iddd in range(len(used_list_f1)):
                    if f1_index==used_list_f1[iddd] and dir1_index==used_list_dir1[iddd]:
                        continue_flag = True
                else:
                    used_list_f1.append(f1_index)
                    used_list_dir1.append(dir1_index)


                if continue_flag:
                    continue

                current_batch, read_complete_flag = read_one_need_from_seq(f1_index, self.seq_len, self.poses, self.range_image_root)
                if not read_complete_flag:
                    continue

                sample_batch, sample_truth, pos_num, neg_num, read_complete_flag = read_one_batch_pos_neg \
                    (f1_index, dir1_index, self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.range_image_root,
                     self.train_overlap, self.overlap_thresh, self.seq_len, self.poses)

                if not read_complete_flag:
                    continue

                use_pos_num = 3
                use_neg_num = 3

                if pos_num >= use_pos_num and neg_num >= use_neg_num:  # 4
                    sample_batch = torch.cat((sample_batch[0:use_pos_num, :, :, :],
                                              sample_batch[pos_num:pos_num + use_neg_num, :, :, :]), dim=0)
                    sample_truth = torch.cat(
                        (sample_truth[0:use_pos_num, :], sample_truth[pos_num:pos_num + use_neg_num, :]), dim=0)
                    pos_num = use_pos_num
                    neg_num = use_neg_num
                elif pos_num >= use_pos_num:
                    sample_batch = torch.cat(
                        (sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:use_pos_num, :], sample_truth[pos_num:, :]), dim=0)
                    pos_num = use_pos_num
                elif neg_num >= use_neg_num:
                    sample_batch = sample_batch[0:pos_num + use_neg_num, :, :, :]
                    sample_truth = sample_truth[0:pos_num + use_neg_num, :]
                    neg_num = use_neg_num


                if neg_num == 0 or pos_num == 0:
                    continue
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                input_batch = torch.cat((current_batch, sample_batch), dim=0)

                input_batch.requires_grad_(True)
                self.amodel.train()
                self.optimizer.zero_grad()

                global_des = self.amodel(input_batch)

                o1, o2, o3 = torch.split(
                    global_des, [1, pos_num, neg_num], dim=0)

                MARGIN_1 = 0.5
                loss = PNV_loss.triplet_loss(o1, o2, o3, MARGIN_1, lazy=False)

                loss.backward()
                self.optimizer.step()
                print(str(used_num), loss)

                if torch.isnan(loss):
                    print(pos_num)
                    print(neg_num)

                loss_each_epoch = loss_each_epoch + loss.item()
                used_num = used_num + 1

            print("epoch {} loss {}".format(i, loss_each_epoch / used_num))
            print("saving weights ...")
            self.scheduler.step()
            self.save_name = "./amodel_seqot"+str(i)+".pth.tar"

            torch.save({
                'epoch': i,
                'state_dict': self.amodel.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
                self.save_name)

            print("Model Saved As " + self.save_name)
            writer1.add_scalar("loss", loss_each_epoch / used_num, global_step=i)



if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    traindata_file = config["training_seqot"]["traindata_file"]
    poses_file = config["training_seqot"]["poses_file"]
    height = config["training_seqot"]["height"]
    width = config["training_seqot"]["width"]
    seqlen = config["training_seqot"]["seqlen"]
    learning_rate = config["training_seqot"]["lr"]
    resume = config["training_seqot"]["resume"]
    pretrained_weights = config["training_seqot"]["weights"]

    range_image_root = config["data_root"]["range_image_database_root"]
    # ============================================================================
    train_handler = trainHandler(height=32, width=900, seqlen=seqlen, lr=learning_rate, resume=resume, pretrained_weights=pretrained_weights,
                                 train_set=traindata_file, poses_file=poses_file, range_image_root=range_image_root)

    train_handler.train()