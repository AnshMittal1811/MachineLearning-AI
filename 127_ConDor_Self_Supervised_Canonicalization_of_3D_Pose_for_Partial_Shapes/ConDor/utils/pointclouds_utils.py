from functools import partial
import vispy
import numpy as np
from scipy.spatial import cKDTree, distance_matrix

import vispy.scene
from vispy.scene import visuals
import skimage
import tensorflow as tf

def tf_sq_distance_matrix_l2(X, Y):
    XY = tf.einsum('bic,bjc->bij', X, Y)
    X2 = tf.reduce_sum(tf.multiply(X, X), axis=-1, keepdims=False)
    Y2 = tf.reduce_sum(tf.multiply(Y, Y), axis=-1, keepdims=False)
    X2 = tf.expand_dims(X2, axis=-1)
    Y2 = tf.expand_dims(Y2, axis=-2)
    return tf.add(tf.subtract(X2, 2.*XY), Y2)


def load_pts(path):
    """takes as input the path to a .pts and returns a list of
	tuples of floats containing the points in in the form:
	[(x_0, y_0, z_0),
	 (x_1, y_1, z_1),
	 ...
	 (x_n, y_n, z_n)]"""
    with open(path) as f:
        rows = [rows.strip() for rows in f]

    """Use the curly braces to find the start and end of the point data"""
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [[float(point) for point in coords] for coords in coords_set]
    return np.array(points, dtype=np.float32)

def uniform_pc_sampling(x, n_samples):
    nv = x.shape[1]
    assert(nv >= n_samples)
    idx = np.arange(nv)
    np.random.shuffle(idx)
    x = np.take(x, idx, axis=1)
    return x[:,0:n_samples, ...]

"""
def uniform_pc_sampling(X, n_samples):
    batch_size = X.shape[0]
    Y = np.zeros(shape=(X.shape[0], n_samples, 3), dtype=np.float32)
    for i in range(batch_size):
        Y[i, ...] = uniform_pc_sampling_(X[i, ...], n_samples)
    return Y
"""

def normalize_point_cloud(arr):
    mean = np.mean(arr, axis=0, keepdims=True)
    c_arr = np.subtract(arr, mean)
    norm = np.linalg.norm(c_arr, axis=-1, keepdims=True)
    max_norm = np.max(norm, axis=0, keepdims=True)
    return np.divide(c_arr , max_norm)

def generate_3d():
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M

def rotate_point_cloud(arr):
    return np.einsum('ij,vj->vi', generate_3d(), arr)

def generate_rot_z():
    x1 = np.random.rand()
    R = np.array([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    return R

def rotate_z(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_matrix = generate_rot_z()
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def generate_rot_y():
    x1 = np.random.rand()
    c = np.cos(2 * np.pi * x1)
    s = np.sin(2 * np.pi * x1)

    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return R

def rotate_y(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_matrix = generate_rot_y()
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_batch(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotations = []
    for k in range(batch_data.shape[0]):
        rotation_matrix = generate_3d()
        rotations.append(rotation_matrix)
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data, rotations

def rotate_pc_normals_batch(batch_data, batch_normals):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_normals = np.zeros(batch_data.shape, dtype=np.float32)
    rotations = []
    for k in range(batch_data.shape[0]):
        rotation_matrix = generate_3d()
        rotations.append(rotation_matrix)
        shape_pc = batch_data[k, ...]
        shape_normals = batch_normals[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_normals[k, ...] = np.dot(shape_normals.reshape((-1, 3)), rotation_matrix)
    return rotated_data, rotated_normals, rotations

def rotate_z_pc_normals_batch(batch_data, batch_normals):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_normals = np.zeros(batch_data.shape, dtype=np.float32)
    rotations = []
    for k in range(batch_data.shape[0]):
        rotation_matrix = generate_rot_z()
        rotations.append(rotation_matrix)
        shape_pc = batch_data[k, ...]
        shape_normals = batch_normals[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_normals[k, ...] = np.dot(shape_normals.reshape((-1, 3)), rotation_matrix)
    return rotated_data, rotated_normals, rotations

def frames_uv(n):
    shape = n.shape
    zero = np.zeros(shape[:-1])
    u0 = np.stack([zero, n[..., 2], -n[..., 1]], axis=-1)
    u1 = np.stack([-n[..., 2], zero, n[..., 0]], axis=-1)
    u2 = np.stack([n[..., 1], -n[..., 0], zero], axis=-1)
    Q = np.stack([u0, u1, u2], axis=-2)
    Q_norm2 = np.multiply(Q, Q)
    Q_norm2 = np.sum(Q_norm2, axis=-1, keepdims=True)
    idx = np.argmax(Q_norm2, axis=-2)
    idx = np.expand_dims(idx, axis=-1)
    u = np.take_along_axis(Q, idx, axis=-2)
    u = u[..., 0, :]
    # normalize u
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(n, u)
    return u, v


def random_batch_scaling_normals(batch_data, normals):
    a = np.random.uniform(low=2. / 3., high=3. / 2., size=[batch_data.shape[0], 1, 3])
    b = np.random.uniform(low=-0.2, high=0.2, size=[batch_data.shape[0], 1, 3])
    u, v = frames_uv(normals)
    u = np.multiply(a, u)
    v = np.multiply(a, v)
    n = np.cross(u, v)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return np.add(np.multiply(a, batch_data), b), n

def random_batch_scaling(batch_data):

    """
    translated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

        shape_pc = batch_data[k, ...]

        translated_data[k, ...] = np.add(np.multiply(shape_pc, xyz1), xyz2)
    """

    # a = 2. / 3. + (3. / 2. - 2. / 3.) * np.random.rand(batch_data.shape[0], 1, 1)
    # a = 3. / 4. + (4. / 3. - 3. / 4.) * np.random.rand(batch_data.shape[0], 1, 1)


    a = np.random.uniform(low=2. / 3., high=3. / 2., size=[batch_data.shape[0], 1, 3])
    b = np.random.uniform(low=-0.2, high=0.2, size=[batch_data.shape[0], 1, 3])
    




    return np.add(np.multiply(a, batch_data), b)
    # return np.multiply(a, np.add(batch_data, b))

def cov_mat(X):
    c = np.mean(X, axis=0, keepdims=True)
    X = np.subtract(X, c)
    sd = np.multiply(X, X)
    sd = np.sum(sd, axis=1, keepdims=True)
    sd = np.sqrt(sd)
    X /= sd

    # cov = np.matmul(X.T, X)

    cov = np.einsum('ij,ik->jk', X, X)
    return cov

def random_perturb(v):
    r = np.random.rand(3)
    r = 0.1*(r / (np.linalg.norm(r) + 0.0001))
    v = v + r
    return v / (np.linalg.norm(v) + 0.0001)

def principal_direction(X):
    cov = cov_mat(X)
    w, v = np.linalg.eigh(cov)
    return v[:, 2]

def eigen_tree_index_(X, depth, redundancy=0):
    nv = X.shape[0]
    if depth == 0:
        return X
    v = principal_direction(X)
    Xv = np.einsum('j,ij->i', v, X)
    idx_v = np.argsort(Xv, axis=0)
    split_idx = np.split(idx_v, 2)
    sp1 = np.take(X, split_idx[0], axis=0)
    sp1 = eigen_tree_index_(sp1, depth-1)
    sp2 = np.take(X, split_idx[1], axis=0)
    sp2 = eigen_tree_index_(sp2, depth-1)
    return np.concatenate([sp1, sp2], axis=0)

def eigen_tree_index_2(X, depth):
    nv = X.shape[0]
    if depth == 0:
        return X
    v = principal_direction(X)
    Xv = np.einsum('j,ij->i', v, X)
    idx_v = np.argsort(Xv, axis=0)
    step = 1./6.
    L = []
    for i in range(4):
        idx_v_i = idx_v[int(step*i*nv):int(step*i*nv)+int(nv/2.)]
        x_i = np.take(X, idx_v_i, axis=0)
        L.append(eigen_tree_index_2(x_i, depth-1))
    return np.concatenate(L, axis=0)

def eigen_tree_index(batch_data, depth):
    nb = batch_data.shape[0]
    for i in range(nb):
        x = batch_data[i, ...]
        batch_data[i, ...] = eigen_tree_index_(x, depth)
    return batch_data


def split_tree(pts, idx, nv, num_cells, num_splits, split_method, split_frac=None):

    cell_size = int(nv / num_cells)

    if split_frac is None:
        split_size = int(cell_size / num_splits)
    else:
        split_size = int(split_frac * cell_size)
        split_size = min(max(split_size, 1), cell_size)


    pts_ = pts[0]
    pts_tmp_ = pts[1]
    idx_ = idx[0]
    idx_tmp_ = idx[1]

    n = 0
    for i in range(0, nv, cell_size):
        pts_i, idx_i = split_method(pts_[i:i+cell_size], num_splits, split_size)
        idx_i = np.take(idx_[i:i+cell_size], idx_i)
        m = idx_i.shape[0]
        pts_tmp_[n:n+m, :] = pts_i
        idx_tmp_[n:n+m] = idx_i
        n += m

    pts[0] = pts_tmp_
    # pts[1] = pts_
    idx[0] = idx_tmp_
    # idx[1] = idx_

    new_nv = split_size*num_splits*num_cells
    new_num_cells = num_splits * num_cells

    return pts, idx, new_nv, new_num_cells


def eigen_split(x, num_splits, split_size):

    nv = x.shape[0]

    v = principal_direction(x)
    # v = np.array([1., 0., 0.])
    # Xv = np.matmul(x, v)

    Xv = np.einsum('j,ij->i', v, x)
    idx_v = np.argsort(Xv, axis=0)
    m = float(nv - split_size)/float(num_splits-1)
    # a_init = int(m*num_splits)
    idx = np.zeros((num_splits*split_size, ), dtype=np.int32)
    pts = np.zeros((num_splits*split_size, 3), dtype=np.int32)

    for i in range(num_splits):
        a = int(m*i)
        b = a + split_size
        idx[i*split_size:(i+1)*split_size] = idx_v[a:b]
        pts[i*split_size:(i+1)*split_size] = np.take(x, idx_v[a:b], axis=0)
    return pts, idx

def xyz_split():
    return 0

def total_nv_(nv_init, num_splits, split_frac):
    depth = len(num_splits)
    num_cells = 1
    nv = nv_init
    cell_size = nv_init
    for i in range(depth):
        num_cells *= num_splits[i]
        split_size = int(split_frac[i] * cell_size)
        split_size = min(max(split_size, 1), split_size)
        cell_size = split_size
        nv = split_size*num_cells
    return nv


def build_tree(X, num_splits, split_method, split_frac=None):
    depth = len(num_splits)

    if split_frac is None:
        split_frac = []
        for i in range(depth):
            split_frac.append(None)

    nv = [X.shape[0]]

    total_nv = total_nv_(nv[0], num_splits, split_frac=split_frac)

    pts0 = np.zeros(shape=(total_nv, 3), dtype=np.float32)
    pts0[:nv[0], :] = X
    pts1 = np.zeros(shape=(total_nv, 3), dtype=np.float32)
    pts = [pts0, pts1]

    idx0 = np.zeros((total_nv,), dtype=np.int32)
    idx0[:nv[0]] = np.arange(nv[0])
    idx1 = np.zeros((total_nv,), dtype=np.int32)
    idx = [idx0, idx1]


    num_cells = [1]
    for i in range(depth):

        pts_, idx_, new_nv, num_cells_ = split_tree(pts, idx, nv[-1], num_cells[-1], num_splits[i],
                                                    split_method, split_frac=split_frac[i])
        pts = pts_
        idx = idx_
        nv.append(new_nv)
        num_cells.append(num_cells[-1] * num_splits[i])


    return pts[0], idx[0], nv, num_cells

"""
def build_tree(X, num_splits, split_method, split_frac=None):
    depth = len(num_splits)
    if num_splits
"""

def tree_1(x):
    pts, idx, nv, num_cells = build_tree(x, [4, 4, 4, 4], eigen_split, split_frac=[0.5, 0.5, 0.5, 0.5])
    return pts


def is_power2(num):
    'states if a number is a power of two'
    return num != 0 and ((num & (num - 1)) == 0)

def binary_tree_pooling(batch_data, k=2, num_points=None):
    nv = batch_data.shape[1]
    if num_points is not None:
        assert(is_power2(num_points) and num_points <= nv)
        k = int(np.rint(np.log(float(nv)/float(num_points))/np.log(2.)))
    return skimage.measure.block_reduce(batch_data, block_size=(1, 2**k, 1), func=np.mean)

def kdtree_index_pc(batch_data):
    nb = batch_data.shape[0]
    for i in range(nb):
        x = batch_data[i, ...]
        T = cKDTree(x)
        batch_data[i, ...] = np.take(x, T.indices, axis=0)
    return batch_data


def kd_tree_index_pairs(x0, x1, y01, y10):
    for i in range(x0.shape[0]):
        x = x0[i, ...]
        T0 = cKDTree(x)
        x0[i, ...] = np.take(x0, T0.indices, axis=0)

        x = x1[i, ...]
        T1 = cKDTree(x)
        x1[i, ...] = np.take(x1, T1.indices, axis=0)

        y01 = np.argsort(T1.indices)[y01[T0.indices]]
        y10 = np.argsort(T0.indices)[y10[T1.indices]]

        return x0, x1, y01, y10

def kdtree_idx(x):
    T = cKDTree(x)
    idx = T.indices
    return idx.astype(np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_kd_tree_wrapper(input):
  idx = tf.numpy_function(kdtree_idx, [input], tf.float32)
  return idx

def tf_kd_tree_idx(x):
    idx = tf.map_fn(fn=tf_kd_tree_wrapper, elems=x)
    idx = tf.cast(idx, tf.int32)
    batch_idx = tf.range(x.shape[0])
    batch_idx = tf.reshape(batch_idx, (x.shape[0], 1))
    batch_idx = tf.tile(batch_idx, (1, x.shape[1]))
    idx = tf.stack([batch_idx, idx], axis=-1)
    return idx




def kdtree_index(X, Y):
    nb = X.shape[0]
    for i in range(nb):
        x = X[i, ...]
        T = cKDTree(x)
        X[i, ...] = np.take(x, T.indices, axis=0)
        y = Y[i, ...]
        Y[i, ...] = np.take(y, T.indices, axis=0)
    return X, Y








def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):

    # sigma *= 2
    # clip *= 2

    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

def normalize_point_cloud_batch(batch_data):
    mean = np.mean(batch_data, axis=1, keepdims=True)
    batch_data = batch_data - mean
    # norm2 = (batch_data ** 2).sum(axis=2)
    norm = np.linalg.norm(batch_data, axis=-1, keepdims=True)
    # sd = np.sqrt(np.mean(norm2))
    sd = np.max(norm, axis=1, keepdims=True)

    return np.divide(batch_data, sd)

def density_noise(x, f, eps=0.1):
    num_points = f.shape[1]
    r = np.random.rand(f.shape[0], num_points, 1)
    f = (1.-eps)*f + eps*r
    mean = np.mean(f, axis=1, keepdims=True)
    f = np.divide(f, mean)
    return f


def pc_normals_batch_preprocess(x, normals, f=None, y=None, proc=None):
    if proc == 'normalize':
        x = normalize_point_cloud_batch(x)
    if proc == 'scale':
        x, normals = random_batch_scaling_normals(x, normals)
    if proc == 'rotate':
        x, normals, m = rotate_pc_normals_batch(x, normals)
    if proc == 'rotate_z':
        x, normals, m = rotate_z_pc_normals_batch(x, normals)
    if proc == 'jitter':
        x = jitter_point_cloud(x, sigma=0.01, clip=0.05)
    if proc == 'noise':
        if f is not None:
            f = density_noise(x, f, eps=0.2)
    if proc == 'kd_tree_idx':
        nb = x.shape[0]
        for i in range(nb):
            xi = x[i, ...]
            T = cKDTree(xi)
            x[i, ...] = np.take(xi, T.indices, axis=0)
            ni = normals[i, ...]
            normals[i, ...] = np.take(ni, T.indices, axis=0)
            if y is not None:
                yi = y[i, ...]
                y[i, ...] = np.take(yi, T.indices, axis=0)
            if f is not None:
                fi = f[i, ...]
                f[i, ...] = np.take(fi, T.indices, axis=0)

    return x, normals, y, f


def np_kd_tree_idx(X):
    batch_size = X.shape[0]
    idx = np.zeros((X.shape[0], X.shape[1]), dtype=np.int32)
    for i in range(batch_size):
        T = cKDTree(X[i, ...])
        idx[i, :] = T.indices
    return idx


def pc_batch_preprocess(x, f=None, y=None, proc=None):
    if proc == 'normalize':
        x = normalize_point_cloud_batch(x)
    if proc == 'scale':
        x = random_batch_scaling(x)
    if proc == 'rotate':
        x, m = rotate_point_cloud_batch(x)
    if proc == 'rotate_z':
        x = rotate_z(x)
    if proc == 'rotate_y':
        x = rotate_y(x)
    if proc == 'jitter':
        x = jitter_point_cloud(x, sigma=0.01, clip=0.05)
    if proc == 'noise':
        if f is not None:
            f = density_noise(x, f, eps=0.2)
    if proc == 'kd_tree_idx':
        if y is not None:
            if y.ndim >= 2:
                x, y = kdtree_index(x, y)
            else:
                x = kdtree_index_pc(x)
                # x = cover_tree_index(x, 4)
        else:
            x = kdtree_index_pc(x)
    if y is None:
        if f is None:
            return x
        else:
            return [x, f]
    else:
        if f is None:
            return x, y
        else:
            return [x, f], y


def pc_pairs_batch_preprocess(X0, X1, y01, y10, proc):
    for p in proc:
        if p == 'normalize':
            X0 = normalize_point_cloud_batch(X0)
            X1 = normalize_point_cloud_batch(X1)
        if p == 'scale':
            X0 = random_batch_scaling(X0)
            X1 = random_batch_scaling(X1)
        if p == 'rotate':
            X0 = rotate_point_cloud_batch(X0)
            X1 = rotate_point_cloud_batch(X1)
        if p == 'jitter':


            sigma = 0.01
            clip = 0.05


            """
            sigma = 0.02
            clip = 0.1
            """

            X0 = jitter_point_cloud(X0, sigma=sigma, clip=clip)
            X1 = jitter_point_cloud(X1, sigma=sigma, clip=clip)
        if p == 'kd_tree_idx':
            X0, X1, y01, y10 = kd_tree_index_pairs(X0, X1, y01, y10)
    # x = np.stack([X0, X1], axis=-1)
    # y = np.stack([y01, y10], axis=-1)
    return X0, X1, y01, y10

"""
def pc_preprocess(x, y, proc):
    if proc == 'normalize':
        x = normalize_point_cloud_batch(x)
    if proc == 'scale':
        x = random_batch_scaling(x)
    if proc == 'rotate':
        x = rotate_point_cloud_batch(x)
    if proc == 'jitter':
        x = jitter_point_cloud(x, sigma=0.01, clip=0.05)
    if proc == 'kd_tree_idx':
        T = cKDTree(x)
        x = np.take(x, T.indices, axis=0)

    return x, y
"""

def sample_faces(vertices, faces, n_samples=10**4):
  """
  Samples point cloud on the surface of the model defined as vectices and
  faces. This function uses vectorized operations so fast at the cost of some
  memory.

  Parameters:
    vertices  - n x 3 matrix
    faces     - n x 3 matrix
    n_samples - positive integer

  Return:
    vertices - point cloud

  Reference :
    [1] Barycentric coordinate system

    \begin{align}
      P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
    \end{align}
  """
  vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
  face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
  face_areas = face_areas / np.sum(face_areas)

  # Sample exactly n_samples. First, oversample points and remove redundant
  # Contributed by Yangyan (yangyan.lee@gmail.com)
  n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
  # floor_num = np.sum(sample_num_per_face) - n_samples
  floor_num = np.sum(n_samples_per_face) - n_samples
  if floor_num > 0:
    indices = np.where(n_samples_per_face > 0)[0]
    floor_indices = np.random.choice(indices, floor_num, replace=True)
    n_samples_per_face[floor_indices] -= 1

  n_samples = np.sum(n_samples_per_face)

  # Create a vector that contains the face indices
  sample_face_idx = np.zeros((n_samples, ), dtype=int)
  acc = 0
  for face_idx, _n_sample in enumerate(n_samples_per_face):
    sample_face_idx[acc: acc + _n_sample] = face_idx
    acc += _n_sample

  r = np.random.rand(n_samples, 2)
  A = vertices[faces[sample_face_idx, 0], :]
  B = vertices[faces[sample_face_idx, 1], :]
  C = vertices[faces[sample_face_idx, 2], :]
  P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
      np.sqrt(r[:, 0:1]) * r[:, 1:] * C

  return P, sample_face_idx, r


def normalize_batch(X):
    c = np.mean(X, axis=1, keepdims=True)
    X = X - c
    n = np.multiply(X, X)
    n = np.sum(n, axis=-1, keepdims=True)
    n = np.sqrt(n)
    n = np.max(n, axis=1, keepdims=True)
    n = np.mean(n, axis=0, keepdims=True)
    X = X / n
    return X

def fps(x, num_points, idx=None):
    nv = x.shape[0]
    # d = distance_matrix(x, x)
    if idx is None:
        idx = np.random.randint(low=0, high=nv-1)
    y = np.zeros(shape=(num_points, 3))
    indices = np.zeros(shape=(num_points, ), dtype=np.int32)
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

def setup_pcl_viewer(X, color=(1, 1, 1, .5), run=False, point_size=5):
    # setup a point cloud viewer using vispy and return a drawing function
    # make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    # create scatter object and fill in the data
    # init_pc = np.random.normal(size=(100, 3), scale=0.2)
    init_pc = X
    scatter = visuals.Markers()
    draw_fn = partial(scatter.set_data, edge_color=None, face_color=color, size=point_size)
    draw_fn(init_pc)
    view.add(scatter)
    # set camera
    view.camera = 'turntable'  # ['turntable','arcball']
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    return draw_fn

class PclViewer():
    def __init__(self, X=None, color=None):

        if X is None:
            self.X = np.array([[0., 0., 0.]], dtype=np.float32)
        else:
            self.X = X
        if color is None:
            self.color = np.array([0.5, 0.5, 0.5, 1.])
        else:
            self.color = color
        # Now let us visualize array 'coords'
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', bgcolor='white')
        self.view = self.canvas.central_widget.add_view()

        # 3D axis
        self.axis = vispy.scene.visuals.XYZAxis(parent=self.view.scene)
        # Scatter plot and line


        # Now, make the markers and the line.
        # Use the parent argument so they are automatically added to the scene.
        """
        self.scatter = vispy.scene.visuals.Markers(pos=self.X,
                                                   edge_color=None,
                                                   face_color=self.color,
                                                   size=5,
                                                   parent=self.view.scene)
        """
        self.scatter = visuals.Markers()

        self.draw_fn = partial(self.scatter.set_data, edge_color=None, face_color=self.color, size=7)

        self.draw_fn(self.X)

        self.view.add(self.scatter)
        """
        self.line = vispy.scene.visuals.Line(pos=self.X,
                                             color=(0., 0., 0., 0.),
                                             width=3,
                                             parent=self.view.scene)
        """
        self.view.camera = 'turntable'
        # self.canvas.show()

    def run(self):
        self.canvas.show()
        vispy.app.run()


    def close(self):
        vispy.app.quit()

    def set_pcl(self, X, color=None):
        if color is not None:
            self.color = color
        self.X = X

        """
        self.scatter.set_data(X,
                         edge_color=None,
                         face_color=self.color,
                         size=5)
        """
        self.draw_fn(self.X)

        # self.view.add(self.scatter)


        # self.line.set_data(pos=X, color=self.color, width=3)


