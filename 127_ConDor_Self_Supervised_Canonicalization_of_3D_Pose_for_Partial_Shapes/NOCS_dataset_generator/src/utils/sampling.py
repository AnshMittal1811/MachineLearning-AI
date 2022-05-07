import numpy as np
from scipy.spatial import cKDTree, distance_matrix
import os
import h5py
import vispy
import vispy.scene
from vispy.scene import visuals
from functools import partial
from scipy.spatial import cKDTree


def fps(x, num_points, idx=None):
    nv = x.shape[0]
    # d = distance_matrix(x, x)
    if idx is None:
        idx = np.random.randint(low=0, high=nv - 1)
    elif idx == 'center':
        c = np.mean(x, axis=0, keepdims=True)
        d = distance_matrix(c, x)
        idx = np.argmax(d)

    y = np.zeros(shape=(num_points, 3))
    indices = np.zeros(shape=(num_points,), dtype=np.int32)
    p = x[np.newaxis, idx, ...]
    dist = distance_matrix(p, x)
    for i in range(num_points):
        y[i, ...] = p
        indices[i] = idx
        d = distance_matrix(p, x)
        dist = np.minimum(d, dist)
        idx = np.argmax(dist)
        p = x[np.newaxis, idx, ...]
    return y, indices




def get_distance_matrix(pts):
    '''Returns the distance matrix

    pts - N x 3
    '''

    
    D = np.mean(np.abs(pts[None, :, :] - pts[:, None, :]), axis = -1)

    return D