from src.dataset.dataloader import ImageData, FaceRawDataset, make_dataset
import numpy as np
import pickle
import argparse
import os
import math
import threading
import scipy.io as sio
import ast
from config import *

train_data_block_names = [f'{BLOCK_DIR}/{i}.pkl' for i in range(NUM_BLOCKS)]


def saveBlock(data_list, worker_id):
    i = 0
    for temp_data in data_list:
        i += 1
        print('worker', worker_id, 'task', i, end='\r')
        temp_data.image = None
        temp_data.pos_map = None
        temp_data.attention_mask = None

        # temp_data.image = temp_data.get_image()
        fp = open(temp_data.image_path, 'rb')
        temp_data.image = fp.read()
        fp.close()
        temp_data.pos_map = temp_data.get_pos_map()

        fp = open(temp_data.attention_path, 'rb')
        temp_data.attention_mask = fp.read()
        fp.close()

    print('saving data block', worker_id)
    f = open(f'{BLOCK_DIR}/{worker_id}.pkl', 'wb')
    pickle.dump(data_list, f)
    f.close()
    for temp_data in data_list:
        temp_data.image = None
        temp_data.pos_map = None
        temp_data.attention_mask = None
    print('data path list saved', worker_id)


def multiSaveBlock(st, ed):
    worker_num = NUM_BLOCKS
    total_task = len(all_data)
    import random
    random.seed(0)
    random.shuffle(all_data)
    jobs = []
    task_per_worker = math.ceil(total_task / worker_num)
    st_idx = [task_per_worker * i for i in range(worker_num)]
    ed_idx = [min(total_task, task_per_worker * (i + 1)) for i in range(worker_num)]
    for i in range(st, ed):
        # temp_data_processor = copy.deepcopy(data_processor)
        p = threading.Thread(target=saveBlock, args=(
            all_data[st_idx[i]:ed_idx[i]], i))
        jobs.append(p)
        p.start()
        print('start ', i)
        if (i - st) % 4 == 3:
            for p in jobs:
                p.join()
        jobs = []
        print('batch end')

    print('all start')


if __name__ == '__main__':
    all_data = []
    if not os.path.exists(BLOCK_DIR):
        os.mkdir(BLOCK_DIR)
    train_dataset = make_dataset(TRAIN_DIR, 'train')
    all_data = train_dataset.train_data
    multiSaveBlock(0, NUM_BLOCKS)
