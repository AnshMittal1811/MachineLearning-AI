import scipy.io
import numpy as np
import json


def load(name, path):
    if name == 'UTD_MHAD':
        arr = scipy.io.loadmat(path)['d_skel']
        new_arr = np.zeros([arr.shape[2], arr.shape[0], arr.shape[1]])
        for i in range(arr.shape[2]):
            for j in range(arr.shape[0]):
                for k in range(arr.shape[1]):
                    new_arr[i][j][k] = arr[j][k][i]
        return new_arr
    elif name == 'HumanAct12':
        return np.load(path, allow_pickle=True)
    elif name == "CMU_Mocap":
        return np.load(path, allow_pickle=True)
    elif name == "Human3.6M":
        return np.load(path, allow_pickle=True)[0::5] # down_sample
    elif name == "NTU":
        return np.load(path, allow_pickle=True)
    elif name == "SHARP":
        poses = np.load(path, allow_pickle=True)
        poses = poses[np.newaxis, ...]
        return poses
