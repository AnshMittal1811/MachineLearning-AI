#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: Generate ground truth file by distance


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('./utils/')
import yaml
import numpy as np
from tools.utils.utils import *

# load config ================================================================
config_filename = '../config/config.yml'
config = yaml.safe_load(open(config_filename))
poses_database = np.load(config["gen_training_index"]["poses_database"])
poses_query = np.load(config["gen_training_index"]["poses_query"])
# ============================================================================

print("How many pose nodes in database: ", poses_database.shape[0])
print("How many pose nodes in query: ", poses_query.shape[0])

all_rows = []
for i in np.arange(0, poses_query.shape[0], 5):
    print(i)
    one_row = []
    for idx in range(0,poses_database.shape[0]):
        if np.linalg.norm(poses_database[idx, :3, -1] - poses_database[i, :3, -1]) < 15 and i!=idx:
            one_row.append(idx)
    all_rows.append(one_row)
    print(str(i) + " ---> ", one_row)
    print("-----------------------------")

print(len(all_rows))
all_rows_array = np.array(all_rows)
np.save("./gt_15dis.npy", all_rows_array)

