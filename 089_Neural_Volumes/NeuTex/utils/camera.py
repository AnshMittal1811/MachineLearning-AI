import numpy as np


def skew_symmetric_matrix(vec):
    assert vec.shape == (3,)
    a1, a2, a3 = vec

    return np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])


def find_common_up(rots):
    mats = []
    for rot in rots:
        x = rot[:, 0]
        z = rot[:, 2]
        mats.append(skew_symmetric_matrix(z) @ skew_symmetric_matrix(x))

    A = np.concatenate(mats, axis=0)
    U, S, Vh = np.linalg.svd(A)
    return Vh[-2]
