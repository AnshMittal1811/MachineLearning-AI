from .global_huffman_code_frequency import global_feature
from .local_length_sequency import length_sequence_all_component
from Encryption_algorithm.encryption_utils import loadEncBit
import numpy as np


if __name__ == '__main__':
    dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size = loadEncBit()   # load encrypted bitstream
    local_feature = length_sequence_all_component(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size)
    np.save("../data/difffeature_matrix.npy", local_feature)
    global_feature = global_feature(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr)
    np.save('../data/huffman_feature.npy', global_feature)
    print('finish save global features.')