import os
import sys
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("../")
# from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
from utils.pointclouds_utils import rotate_point_cloud, normalize_point_cloud
import pandas as pd
from network_utils.pooling import kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_, kd_pooling_1d

# from data_providers.provider import Provider

# from pyntcloud import PyntCloud
from scipy.spatial import cKDTree

SAMPLING_BIN = os.path.join(BASE_DIR, 'third_party/mesh_sampling/build/pcsample')

SAMPLING_POINT_NUM = 2048
SAMPLING_LEAF_SIZE = 0.005

MODELNET40_PATH = '../datasets/modelnet40'


"""
def export_ply(pc, filename):
	vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	for i in range(pc.shape[0]):
		vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
	ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
	ply_out.write(filename)

# Sample points on the obj shape
def get_sampling_command(obj_filename, ply_filename):
    cmd = SAMPLING_BIN + ' ' + obj_filename
    cmd += ' ' + ply_filename
    cmd += ' -n_samples %d ' % SAMPLING_POINT_NUM
    cmd += ' -leaf_size %f ' % SAMPLING_LEAF_SIZE
    return cmd

# --------------------------------------------------------------
# Following are the helper functions to load MODELNET40 shapes
# --------------------------------------------------------------

# Read in the list of categories in MODELNET40
def get_category_names():
    shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

# Return all the filepaths for the shapes in MODELNET40
def get_obj_filenames():
    obj_filelist_file = os.path.join(MODELNET40_PATH, 'filelist.txt')
    obj_filenames = [os.path.join(MODELNET40_PATH, line.rstrip()) for line in open(obj_filelist_file)]
    print('Got %d obj files in modelnet40.' % len(obj_filenames))
    return obj_filenames

# Helper function to create the father folder and all subdir folders if not exist
def batch_mkdir(output_folder, subdir_list):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(output_folder, subdir)):
            os.mkdir(os.path.join(output_folder, subdir))

"""
# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal,
		data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5_data(h5_filename, data,  data_dtype='float32'):
    h5_fout = h5py.File(h5_filename, mode='w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.close()

def save_h5_upsampling(h5_filename, input, downsampled, upsampled, data_dtype='float32'):
    h5_fout = h5py.File(h5_filename, mode='w')
    h5_fout.create_dataset(
        'input', data=input,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'downsampled', data=downsampled,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'upsampled', data=upsampled,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.close()

# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def save_h5_datset(h5_filename, name, data, dtype):
    h5_fout = h5py.File(h5_filename)
    for i in range(len(name)):
        if dtype[i] == 'float32':
            compression_opts = 4
        elif dtype[i] == 'uint8':
            compression_opts = 1
        else:
            compression_opts = 0
        h5_fout.create_dataset(
                name[i], data=data[i],
                compression='gzip', compression_opts=compression_opts,
                dtype=dtype[i])
    h5_fout.close()

# Read numpy array data and label from h5_filename
def load_h5_data(h5_filename):
    f = h5py.File(h5_filename, mode='r')
    data = f['data'][:]
    return data

def load_h5_data_files(data_path, files_list_path):
    files_list = [line.rstrip() for line in open(files_list_path)]
    data = []
    for i in range(len(files_list)):
        data_ = load_h5_data(os.path.join(data_path, files_list[i]))
        data.append(data_)
    data = np.concatenate(data, axis=0)
    return data

def load_h5_data_files_new(data_path, files_list):
    files_list = [ os.path.join(data_path, category_file) for category_file in files_list]
    print(files_list)
    data = []
    for i in range(len(files_list)):
        data_ = load_h5_data(os.path.join(data_path, files_list[i]))
        data.append(data_)
    data = np.concatenate(data, axis=0)
    return data

def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename, mode='r')
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return data, label, normal

def load_h5_normals_files(data_path, files_list_path):
    files_list = [line.rstrip() for line in open(files_list_path)]
    data = []
    labels = []
    normals = []

    for i in range(len(files_list)):
        data_, labels_, normals_ = load_h5_data_label_normal(os.path.join(data_path, files_list[i]))
        data.append(data_)
        labels.append(labels_)
        normals.append(normals_)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    normals = np.concatenate(normals, axis=0)
    return data, labels, normals

# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_files(data_path, files_list_path):
    files_list = [line.rstrip() for line in open(files_list_path)]
    data = []
    labels = []
    for i in range(len(files_list)):
        data_, labels_ = load_h5(os.path.join(data_path, files_list[i]))
        data.append(data_)
        labels.append(labels_)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels


# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files
# ----------------------------------------------------------------

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Make up rows for Nxk array
# Input Pad is 'edge' or 'constant'
def pad_arr_rows(arr, row, pad='edge'):
    assert(len(arr.shape) == 2)
    assert(arr.shape[0] <= row)
    assert(pad == 'edge' or pad == 'constant')
    # arr = np.random.shuffle(arr)
    if arr.shape[0] == row:
        return arr
    if pad == 'edge':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'edge')
    if pad == 'constant':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'constant', (0, 0))


def hdf5_seg_dataset(source_dir, h5_filename, files, NV, rotate):
    n_files = len(files)
    data = np.zeros(shape=(n_files, NV, 3), dtype=np.float32)
    labels = np.zeros(shape=(n_files, NV), dtype=np.int32)

    for i in range(len(files)):
        print(i)
        path = os.path.join(source_dir, files[i][11:]) + '.txt'
        x = np.loadtxt(path)

        nv_x = x.shape[0]

        if nv_x < NV:
            n = int(2*NV/nv_x)
            x = np.repeat(x, repeats=n, axis=0)

        idx = np.arange(NV, dtype=np.int32)
        np.random.shuffle(idx)

        x = np.take(x, indices=idx, axis=0)

        x = x[0:NV, ...]

        xyz = normalize_point_cloud(x[:, 0:3])
        x_labels = x[:, -1]
        if rotate:
            xyz = rotate_point_cloud(xyz)

        data[i, ...] = xyz
        labels[i, ...] = x_labels
    save_h5(h5_filename, data, labels, 'float32', 'uint8')

def sample_points(arr, num_points):

    """
    if num_points > arr.shape[0]:
        return pad_arr_rows(arr, row=num_points)
    else:
        return arr[0:num_points, ...]
    """
    if num_points > arr.shape[0]:
        arr = np.repeat(arr, int(float(num_points)/float(arr.shape[0]))+1, axis=0)
    np.random.shuffle(arr)
    return arr[0:num_points, ...]

def load_dense_matrix(path_file, d_type=np.float32):
    out = np.genfromtxt(fname=path_file, dtype=d_type, delimiter=' ')
    if np.ndim(out) == 1:
        out = np.expand_dims(out, axis=-1)
    # out = np.loadtxt(path_file, delimiter=' ', dtype=d_type)
    # out = np.fromfile(path_file, dtype=d_type, sep=' ')
    # t = pd.read_csv(path_file, dtype=d_type, header=None)
    # out = t.values
    return out


def read_off(file):
    file = open(file, 'r')
    line = file.readline().strip()
    if 'OFF' != line:
        line = line[3:]
        # raise ('Not a valid OFF header')
    else:
        line = file.readline().strip()

    n_verts, n_faces, n_dontknow = tuple([int(s) for s in line.split(' ')])
    verts = []

    for i_vert in range(n_verts):
        verts.append(np.array([float(s) for s in file.readline().strip().split(' ')]))
    verts = np.stack(verts)


    faces = []
    for i_face in range(n_faces):
        faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
    faces = np.stack(faces).astype(np.int32)

    return verts, faces


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
    P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + np.sqrt(r[:,0:1]) * r[:,1:] * C
    return P


def np_mat_to_pandas(mat):
    return pd.DataFrame({'x': mat[:, 0], 'y': mat[:, 1], 'z': mat[:, 2]})

"""
def uniform_mesh_sampling(V, F, num_samples=2048, grid_res=64, mesh_samples=10**4):
    X_ = sample_faces(V, F, n_samples=mesh_samples)
    cloud = np_mat_to_pandas(X_)
    cloud = PyntCloud(cloud)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=grid_res, n_y=grid_res, n_z=grid_res)
    # new_cloud = cloud.get_sample("voxelgrid_nearest", voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
    new_cloud = cloud.get_sample("voxelgrid_centroids", voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
    X = new_cloud.points.as_matrix()
    X = sample_points(X, num_samples)
    return X
"""

def read_and_sample_off(file_path, num_samples=2048):
    V, F = read_off(file_path)
    return uniform_mesh_sampling(V, F, num_samples)


def nn_correspondance(X, Y):
    Ty = cKDTree(Y)
    _, idx = Ty.query(X)
    return idx

def lexicographic_ordering(x):
    m = tf.reduce_min(x, axis=1, keepdims=True)
    y = tf.subtract(x, m)
    M = tf.reduce_max(y, axis=1, keepdims=True)
    y = tf.multiply(M[..., 2]*M[..., 1], y[..., 0]) + tf.multiply(M[..., 2], y[..., 1]) + y[..., 2]
    batch_idx = tf.range(y.shape[0])
    batch_idx = tf.expand_dims(batch_idx, axis=-1)
    batch_idx = tf.tile(batch_idx, multiples=(1, y.shape[1]))
    idx = tf.argsort(y, axis=1)
    idx = tf.stack([batch_idx, idx], axis=-1)
    return tf.gather_nd(x, idx)


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


def tf_random_rotation(shape):
    if isinstance(shape, int):
        shape = [shape]

    batch_size = shape[0]
    t = tf.random.uniform(shape + [3], minval=0., maxval=1.)
    c1 = tf.cos(2 * np.pi * t[:, 0])
    s1 = tf.sin(2 * np.pi * t[:, 0])

    c2 = tf.cos(2 * np.pi * t[:, 1])
    s2 = tf.sin(2 * np.pi * t[:, 1])

    z = tf.zeros(shape)
    o = tf.ones(shape)

    R = tf.stack([c1, s1, z, -s1, c1, z, z, z, o], axis=-1)
    R = tf.reshape(R, shape + [3, 3])

    v1 = tf.sqrt(t[:, -1])
    v3 = tf.sqrt(1-t[:, -1])
    v = tf.stack([c2 * v1, s2 * v1, v3], axis=-1)
    H = tf.tile(tf.expand_dims(tf.eye(3), axis=0), (batch_size, 1, 1)) - 2.* tf.einsum('bi,bj->bij', v, v)
    M = -tf.einsum('bij,bjk->bik', H, R)
    return M

def tf_random_rotate(x, return_rotation = False):
    R = tf_random_rotation(x.shape[0])
    if return_rotation:
        return tf.einsum('bij,bpj->bpi', R, x), R
    return tf.einsum('bij,bpj->bpi', R, x)

def tf_center(x, return_center = False):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    
    if return_center:
        return tf.subtract(x, c), c 
    return tf.subtract(x, c)

def tf_dist(x, y):
    d = tf.subtract(x, y)
    d = tf.multiply(d, d)
    d = tf.reduce_sum(d, axis=-1, keepdims=False)
    d = tf.sqrt(d + 0.00001)
    return tf.reduce_mean(d)

def var_normalize(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    x = tf.subtract(x, c)
    n2 = tf.multiply(x, x)
    n2 = tf.reduce_sum(n2, axis=-1, keepdims=True)
    n2 = tf.reduce_mean(n2, axis=1, keepdims=True)
    sigma = tf.sqrt(n2 + 0.000000001)
    x = tf.divide(x, sigma)
    return x

def registration(x, y):
    x = var_normalize(x)
    y = var_normalize(y)
    xyt = tf.einsum('bvi,bvj->bij', x, y)
    s, u, v = tf.linalg.svd(xyt)
    R = tf.matmul(v, u, transpose_b=True)
    return R



def compute_centroids(points, capsules):
    return tf.einsum('bij,bic->bcj', points, capsules)

def tf_random_dir(shape):
    if isinstance(shape, int):
        shape = [shape]
    t = tf.random.uniform(shape + [3], minval=-1., maxval=1.)
    return tf.linalg.l2_normalize(t, axis=-1)

def partial_shapes(x, num_points_arg = None):
    num_points = x.shape[1]
    if num_points_arg is not None:
        num_points_partial = num_points_arg
    else:
        num_points_partial = num_points // 2
    
    batch_size = x.shape[0]
    v = tf_random_dir(batch_size)
    vx = tf.einsum('bj,bpj->bp', v, x)
    x_partial_idx = tf.argsort(vx, axis=-1)
    x_partial_idx = x_partial_idx[:, :num_points_partial]
    batch_idx = tf.expand_dims(tf.range(batch_size), axis=1)
    batch_idx = tf.tile(batch_idx, (1, num_points_partial))
    x_partial_idx = tf.stack([batch_idx, x_partial_idx], axis=-1)
    x_partial = tf.gather_nd(x, x_partial_idx)
    x_partial, kd_idx, kd_idx_inv = kdtree_indexing_(x_partial)
    return x_partial, x_partial_idx, kd_idx_inv
