import numpy as np

OFFSET_FIX_RATE = 1.5e-5
POSMAP_FIX_RATE = 280.0

NET = 'SADRN'

DATALOADER_WORKER_NUM = 8
TOTAL_EPOCH = 35
START_EPOCH = 0
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4

PRETAINED_MODEL=None
TRAIN_DIR = ['data/dataset/300W_LP_crop', 'data/dataset/Extra_LP']
VAL_DIR = ['data/dataset/AFLW2000_crop']
MODEL_SAVE_DIR = 'data/saved_model'

FACE_UVM_LOSS_RATE = 0
OFFSET_UVM_LOSS_RATE = 0.5
KPT_UVM_LOSS_RATE = 1
ATT_LOSS_RATE = 0.05
SMOOTH_LOSS_RATE = 0.002

DATA_TYPE = 'IPOA'

# block
NUM_BLOCKS = 200
BLOCK_DIR = 'data/dataset/train_blocks/'
NUM_BLOCK_PER_PART = 1
NUM_BLOCK_THREAD = 1
