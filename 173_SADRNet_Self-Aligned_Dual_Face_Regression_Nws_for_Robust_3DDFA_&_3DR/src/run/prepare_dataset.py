from src.dataset.prepare import *
from src.dataset.build_blocks import make_dataset,multiSaveBlock
import scipy.io as sio

if __name__ == '__main__':
    # translate mesh data to uv map data
    multi_process(worker_num=WORKER_NUM, input_dir='data/packs/AFLW2000', output_dir='data/dataset/AFLW2000_crop')
    multi_process(worker_num=WORKER_NUM, input_dir='data/packs/300W_LP', output_dir='data/dataset/300W_LP_crop')
    run_mean_posmap()

    # compress training files into data blocks for faster loading
    all_data = []
    if not os.path.exists(BLOCK_DIR):
        os.mkdir(BLOCK_DIR)
    train_dataset = make_dataset(TRAIN_DIR, 'train')
    all_data = train_dataset.train_data
    multiSaveBlock(0, NUM_BLOCKS)
