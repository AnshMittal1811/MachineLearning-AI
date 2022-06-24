import numpy as np


def Tnorm_VnormC(normt, tri, ntri, nver):
    """
    from matlab TnormVnormC
    :param normt:
    :param tri:
    :param ntri:
    :param nver:
    :return:
    """
    normv = np.zeros((nver, 3))
    for i in range(ntri):
        pt = tri[i]
        for j in range(3):
            for k in range(3):
                normv[pt[j]][k] += normt[i][k]
    return normv


def NormDirection(vertex, tri):
    """
    from matlab
    :param vertex:
    :param tri:
    :return:
    """
    pt1 = vertex[tri[:, 0], :]
    pt2 = vertex[tri[:, 1], :]
    pt3 = vertex[tri[:, 2], :]
    n_tri = np.cross(pt1 - pt2, pt1 - pt3)

    N = np.zeros((vertex.shape[0], 3))
    N = Tnorm_VnormC(n_tri, tri, tri.shape[0], vertex.shape[0])
    mag = np.sum(N * N, axis=1)
    co = np.nonzero(mag == 0)
    mag[co] = 1
    N[co, 0] = np.ones((len(co)))
    mag2 = np.tile(mag, (3, 1)).T
    N = N / np.sqrt(mag2)
    N = -N
    return N



