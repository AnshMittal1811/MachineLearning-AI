#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: Generate training indices


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('./utils/')
import yaml
import numpy as np
from tools.utils.com_overlap_yaw import com_overlap_yaw
from tools.utils.utils import *


# load config ================================================================
config_filename = '../config/config.yml'
config = yaml.safe_load(open(config_filename))
poses_database = np.load(config["gen_training_index"]["poses_database"])
poses_query = np.load(config["gen_training_index"]["poses_query"])
scan_folder = config["gen_training_index"]["scan_database_root"]
# ============================================================================


print("How many pose nodes in database: ", poses_database.shape[0])
chosen_idx = np.arange(0,poses_database.shape[0],5)
poses_database = poses_database[chosen_idx, :, :]
print("How many pose nodes in downsampled database: ", poses_database.shape[0])
scan_paths_all = load_files(scan_folder)
scan_paths =[]
for idx in chosen_idx:
    scan_paths.append(scan_paths_all[idx])
training_tuple_list = []
bin_interval = [1.1, 0.7, 0.5, 0.3, 0.1, 0]
for i in range(len(scan_paths)):
    print("\nProcessing " + str(i) + "/" + str(len(scan_paths)) + "-------->")
    scan_paths_this_frame = scan_paths[i:]
    poses_new_this_frame = poses_database[i:]
    ground_truth_mapping = com_overlap_yaw(scan_paths_this_frame, poses_new_this_frame, frame_idx=0, leg_output_width=360)
    for m in range(ground_truth_mapping.shape[0]):
        one_row = []
        idx1 = str(int(i)).zfill(6)
        idx2 = str(int(i + ground_truth_mapping[m, 1])).zfill(6)
        one_row.append(idx1)
        one_row.append(idx2)
        one_row.append(ground_truth_mapping[m, 2])
        training_tuple_list.append(one_row)
normalized_array = np.array(training_tuple_list)
print("\n Saving training indices without nomalization ...")
np.save("./no_normalized_data.npy", normalized_array)


################## normalize training data


no_normalize_data = np.load("./no_normalized_data.npy", allow_pickle="True")
print("How many training indices: ", no_normalize_data.shape[0])
for i in range(no_normalize_data.shape[0]):
    no_normalize_data[i,0] = str( int(no_normalize_data[i,0])*5 ).zfill(6)
    no_normalize_data[i,1] = str( int(no_normalize_data[i,1])*5 ).zfill(6)
normalize_data = []
for i in range(no_normalize_data.shape[0]):
    if abs(int(no_normalize_data[i, 0]) - int(no_normalize_data[i, 1])) > 1000 and no_normalize_data[i, 2].astype(float) > 0.3:
        normalize_data.append(no_normalize_data[i, :])
        continue
print("Number of long-term loop closing: ", len(normalize_data))
normalize_data_bin1 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<1.1) & (no_normalize_data[:,-1].astype(float)>=0.7)]
normalize_data_bin2 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<0.7) & (no_normalize_data[:,-1].astype(float)>=0.5)]
normalize_data_bin3 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<0.5) & (no_normalize_data[:,-1].astype(float)>=0.3)]
normalize_data_bin4 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<0.3) & (no_normalize_data[:,-1].astype(float)>=0.1)]
normalize_data_bin5 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<0.1) & (no_normalize_data[:,-1].astype(float)>=0.0)]
print("1.0~0.7: ", normalize_data_bin1.shape)
print("0.7~0.5: ", normalize_data_bin2.shape)
print("0.5~0.3: ", normalize_data_bin3.shape)
print("0.3~0.1: ", normalize_data_bin4.shape)
print("0.1~0.0: ", normalize_data_bin5.shape)
min_bin = 10000
min_bin_pos = 15000
min_bin_neg = 30000
chosen_idx = np.random.randint(0,normalize_data_bin1.shape[0],min_bin_pos)
normalize_data_bin1_chosen = normalize_data_bin1[chosen_idx,:]
chosen_idx = np.random.randint(0,normalize_data_bin2.shape[0],min_bin_pos)
normalize_data_bin2_chosen = normalize_data_bin2[chosen_idx,:]
chosen_idx = np.random.randint(0,normalize_data_bin3.shape[0],min_bin_pos)
normalize_data_bin3_chosen = normalize_data_bin3[chosen_idx,:]
chosen_idx = np.random.randint(0,normalize_data_bin4.shape[0],min_bin_neg)
normalize_data_bin4_chosen = normalize_data_bin4[chosen_idx,:]
chosen_idx = np.random.randint(0,normalize_data_bin5.shape[0],min_bin_neg)
normalize_data_bin5_chosen = normalize_data_bin5[chosen_idx,:]
chosen_idx = np.random.randint(0,np.array(normalize_data).shape[0],min_bin)
normalize_data_bin6_chosen = np.array(normalize_data)[chosen_idx,:]

normalize_data_chosen = np.concatenate((normalize_data_bin1_chosen,normalize_data_bin2_chosen,normalize_data_bin3_chosen,
                                        normalize_data_bin4_chosen, normalize_data_bin5_chosen,normalize_data_bin6_chosen), axis=0)
print("How many training indices: ", normalize_data_chosen.shape[0])
np.save("./normalized_data.npy", normalize_data_chosen)


