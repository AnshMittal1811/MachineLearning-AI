import numpy as np
import tensorflow as tf
from utils.pointclouds_utils import distance_matrix


def tf_sq_distance_matrix(x, y):
    # compute distance mat
    target = y
    source = x
    r0 = tf.multiply(target, target)
    r0 = tf.reduce_sum(r0, axis=-1, keepdims=True)
    r1 = tf.multiply(source, source)
    r1 = tf.reduce_sum(r1, axis=-1, keepdims=True)
    perm = list(range(len(list(r1.shape))))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    r1 = tf.transpose(r1, perm)
    sq_distance_mat = r0 - 2. * tf.matmul(target, tf.transpose(source, perm)) + r1
    return sq_distance_mat


def tf_patches_idx(sq_dist_mat, patch_size):
    _, patches_idx = tf.nn.top_k(-sq_dist_mat, k=patch_size)
    return patches_idx


def np_polyhedrons(poly):


    C0 = 3 * np.sqrt(2) / 4
    C1 = 9 * np.sqrt(2) / 8

    tetrakis_hexahedron = np.array([[0.0, 0.0, C1],
                                    [0.0, 0.0, -C1],
                                    [C1, 0.0, 0.0],
                                    [-C1, 0.0, 0.0],
                                    [0.0, C1, 0.0],
                                    [0.0, -C1, 0.0],
                                    [C0, C0, C0],
                                    [C0, C0, -C0],
                                    [C0, -C0, C0],
                                    [C0, -C0, -C0],
                                    [-C0, C0, C0],
                                    [-C0, C0, -C0],
                                    [-C0, -C0, C0],
                                    [-C0, -C0, -C0]], dtype=np.float32)

    C0 = (1 + np.sqrt(5)) / 4
    C1 = (3 + np.sqrt(5)) / 4

    regular_dodecahedron = np.array([[0.0, 0.5, C1], [0.0, 0.5, -C1], [0.0, -0.5, C1], [0.0, -0.5, -C1],
                      [C1, 0.0, 0.5], [C1, 0.0, -0.5], [-C1, 0.0, 0.5], [-C1, 0.0, -0.5],
                      [0.5, C1, 0.0], [0.5, -C1, 0.0], [-0.5, C1, 0.0], [-0.5, -C1, 0.0],
                      [C0, C0, C0], [C0, C0, -C0], [C0, -C0, C0], [C0, -C0, -C0],
                      [-C0, C0, C0], [-C0, C0, -C0], [-C0, -C0, C0], [-C0, -C0, -C0]], dtype=np.float32)

    C0 = 3 * (np.sqrt(5) - 1) / 4
    C1 = 9 * (9 + np.sqrt(5)) / 76
    C2 = 9 * (7 + 5 * np.sqrt(5)) / 76
    C3 = 3 * (1 + np.sqrt(5)) / 4

    pentakis_dodecahedron = np.array([[0.0, C0, C3], [0.0, C0, -C3], [0.0, -C0, C3], [0.0, -C0, -C3],
                      [C3, 0.0, C0], [C3, 0.0, -C0], [-C3, 0.0, C0], [-C3, 0.0, -C0],
                      [C0, C3, 0.0], [C0, -C3, 0.0], [-C0, C3, 0.0], [-C0, -C3, 0.0],
                      [C1, 0.0, C2], [C1, 0.0, -C2], [-C1, 0.0, C2], [-C1, 0.0, -C2],
                      [C2, C1, 0.0], [C2, -C1, 0.0], [-C2, C1, 0.0], [-C2, -C1, 0.0],
                      [0.0, C2, C1], [0.0, C2, -C1], [0.0, -C2, C1], [0.0, -C2, -C1],
                      [1.5, 1.5, 1.5], [1.5, 1.5, -1.5], [1.5, -1.5, 1.5], [1.5, -1.5, -1.5],
                      [-1.5, 1.5, 1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5], [-1.5, -1.5, -1.5]],
                     dtype=np.float)


    C0 = 3 * (15 + np.sqrt(5)) / 44
    C1 = (5 - np.sqrt(5)) / 2
    C2 = 3 * (5 + 4 * np.sqrt(5)) / 22
    C3 = 3 * (5 + np.sqrt(5)) / 10
    C4 = np.sqrt(5)
    C5 = (75 + 27 * np.sqrt(5)) / 44
    C6 = (15 + 9 * np.sqrt(5)) / 10
    C7 = (5 + np.sqrt(5)) / 2
    C8 = 3 * (5 + 4 * np.sqrt(5)) / 11

    disdyakis_triacontahedron = np.array([[0.0, 0.0, C8], [0.0, 0.0, -C8], [C8, 0.0, 0.0], [-C8, 0.0, 0.0],
                                          [0.0, C8, 0.0], [0.0, -C8, 0.0], [0.0, C1, C7], [0.0, C1, -C7],
                                          [0.0, -C1, C7], [0.0, -C1, -C7], [C7, 0.0, C1], [C7, 0.0, -C1],
                                          [-C7, 0.0, C1], [-C7, 0.0, -C1], [C1, C7, 0.0], [C1, -C7, 0.0],
                                          [-C1, C7, 0.0], [-C1, -C7, 0.0], [C3, 0.0, C6], [C3, 0.0, -C6],
                                          [-C3, 0.0, C6], [-C3, 0.0, -C6], [C6, C3, 0.0], [C6, -C3, 0.0],
                                          [-C6, C3, 0.0], [-C6, -C3, 0.0], [0.0, C6, C3], [0.0, C6, -C3],
                                          [0.0, -C6, C3], [0.0, -C6, -C3], [C0, C2, C5], [C0, C2, -C5],
                                          [C0, -C2, C5], [C0, -C2, -C5], [-C0, C2, C5], [-C0, C2, -C5],
                                          [-C0, -C2, C5], [-C0, -C2, -C5], [C5, C0, C2], [C5, C0, -C2],
                                          [C5, -C0, C2], [C5, -C0, -C2], [-C5, C0, C2], [-C5, C0, -C2],
                                          [-C5, -C0, C2], [-C5, -C0, -C2], [C2, C5, C0], [C2, C5, -C0],
                                          [C2, -C5, C0], [C2, -C5, -C0], [-C2, C5, C0], [-C2, C5, -C0],
                                          [-C2, -C5, C0], [-C2, -C5, -C0], [C4, C4, C4], [C4, C4, -C4],
                                          [C4, -C4, C4], [C4, -C4, -C4], [-C4, C4, C4], [-C4, C4, -C4],
                                          [-C4, -C4, C4], [-C4, -C4, -C4]], dtype=np.float32)

    P = {'tetrakis_hexahedron':tetrakis_hexahedron,
         'regular_dodecahedron':regular_dodecahedron,
         'pentakis_dodecahedron':pentakis_dodecahedron,
         'disdyakis_triacontahedron':disdyakis_triacontahedron}

    p = P[poly]
    c = np.mean(p, axis=0, keepdims=True)
    p = np.subtract(p, c)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    p = np.divide(p, n)
    return p

def tf_polyhedrons(poly):
    return tf.convert_to_tensor(np_polyhedrons(poly), dtype=tf.float32)

def cross_prod_matrix(v):
    shape = list(v.shape)
    if len(shape) == 1:
        z = 0
    else:
        z = tf.zeros(shape[:-1])
    V = tf.stack([z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z], axis=-1)
    return tf.reshape(V, shape[:-1] + [3, 3])


def S2_transport_matrix(u, v):
    w = cross_prod_matrix(tf.linalg.cross(u, v))
    w2 = tf.matmul(w, w)
    shape = list(w.shape)
    d = tf.multiply(u, v)
    d = tf.reduce_sum(d, axis=-1, keepdims=False)
    d = tf.reshape(d, shape[:-2]+[1, 1])
    I = tf.reshape(tf.eye(3), [1]*(len(shape)-2)+[3, 3])
    R = tf.add(I, w)
    R = tf.add(R, tf.divide(w2, d + 1.0000001))
    I = tf.tile(I, shape[:-2] + [1, 1])
    return tf.where(d > 0.001, R, I)


def SO3_sampling_from_S2(base, k, bundle_shape=False):
    # center and normalize S2 samples just in case
    c = tf.reduce_mean(base, axis=0)
    base = tf.subtract(base, c)
    base = tf.linalg.l2_normalize(base, axis=1)
    # propagate frame from north pole using parallel transport
    t = (2*np.pi/k) * tf.range(k, dtype=tf.float32)
    c = tf.cos(t)
    s = tf.sin(t)
    z = tf.zeros(t.shape)
    f = tf.stack([c, s, z], axis=-1)
    n = tf.constant([0, 0, 1], dtype=tf.float32)
    n = tf.expand_dims(n, axis=0)
    n = tf.tile(n, [base.shape[0], 1])
    R = S2_transport_matrix(n, base)
    v = tf.einsum('vij,dj->vdi', R, f)
    u = tf.tile(tf.expand_dims(base, axis=1), [1, k, 1])
    w = tf.linalg.cross(u, v)
    f = tf.stack([u, v, w], axis=-1)
    if not bundle_shape:
        f = tf.reshape(f, (-1, 3, 3))
    return f

def cos_distance_matrix(U, V):
    U = np.expand_dims(U, axis=1)
    V = tf.expand_dims(V, axis=0)
    d = np.multiply(U, V)
    d = -np.sum(d, axis=-1, keepdims=False)
    return d

def fps(x, num_points, idx=None, distance_fn=distance_matrix):
    nv = x.shape[0]
    # d = distance_matrix(x, x)
    if idx is None:
        idx = np.random.randint(low=0, high=nv-1)
    y = np.zeros(shape=(num_points, x.shape[-1]))
    indices = np.zeros(shape=(num_points, ), dtype=np.int32)
    p = x[np.newaxis, idx, ...]
    dist = distance_fn(p, x)
    for i in range(num_points):
        y[i, ...] = p
        indices[i] = idx
        d = distance_fn(p, x)
        dist = np.minimum(d, dist)
        idx = np.argmax(dist)
        p = x[np.newaxis, idx, ...]
    return y, indices

"""
def hyper_sphere_fps(d, num_samples, num_samples_from=10000):
    samples = np.random.rand((num_samples_from, d+1))
    norm = np.linalg.norm(samples, axis=-1)
    norm = np.expand_dims(norm, axis=-1)
    o = np.full(1./np.sqrt(d))
    samples = np.where(norm > 0.0001, np.divide(samples, norm), o)
    fps(x, num_points, idx=None)
"""



def S2_fps(num_samples, res=100):
    theta = (np.pi/res) * np.arange(1.*res)
    phi = (np.pi/res) * np.arange(2.*res)
    ct = np.expand_dims(np.cos(theta), axis=1)
    tiles = [1]*len(ct.shape)
    tiles[1] = 2*res
    ct = np.tile(ct, tiles)
    st = np.expand_dims(np.sin(theta), axis=1)
    cp = np.expand_dims(np.cos(phi), axis=0)
    sp = np.expand_dims(np.sin(phi), axis=0)
    stcp = np.multiply(st, cp)
    stsp = np.multiply(st, sp)
    points = np.stack([stcp, stsp, ct], axis=-1)
    points = np.reshape(points, (2*res*res, 3))
    samples, _ = fps(points, num_samples, idx=0, distance_fn=cos_distance_matrix)
    return samples

def tf_S2_fps(num_samples, res=100):
    return tf.convert_to_tensor(S2_fps(num_samples, res=res), dtype=tf.float32)

def rot_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    o = np.ones(theta.shape)
    z = np.zeros(theta.shape)
    R = np.stack([c, -s, z, s, c, z, z, z, o], axis=-1)
    return np.reshape(R, list(theta.shape) + [3, 3])

def rot_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    o = np.ones(theta.shape)
    z = np.zeros(theta.shape)
    R = np.stack([c, z, s, z, o, z, -s, z, c], axis=-1)
    return np.reshape(R, list(theta.shape) + [3, 3])

def SO3_fps(num_samples, res=20):
    a = (np.pi/res) * np.arange(2.*res)
    b = (np.pi/res) * np.arange(1. * res)
    c = (np.pi/res) * np.arange(2. * res)
    Ra = rot_z(a)
    Rb = rot_y(b)
    Rc = rot_z(c)

    # R = np.matmul(np.matmul(Ra, Rb), Rc)
    R = np.einsum('aij,bjk->abik', Ra, Rb)
    R = np.einsum('abij,cjk->abcik', R, Rc)
    # R = np.reshape(R, (4 * res ** 3, 3, 3))


    R = np.reshape(R, (4*res**3, 9))
    # R, _ = fps(R, num_samples, idx=0, distance_fn=cos_distance_matrix)
    R, _ = fps(R, num_samples, idx=0, distance_fn=distance_matrix)
    R = np.reshape(R, (-1, 3, 3))

    return R

def tf_SO3_fps(num_samples, res=20):
    return tf.convert_to_tensor(SO3_fps(num_samples=num_samples, res=res), dtype=tf.float32)
