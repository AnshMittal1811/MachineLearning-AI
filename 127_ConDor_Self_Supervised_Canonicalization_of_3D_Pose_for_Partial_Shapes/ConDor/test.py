import tensorflow as tf
from spherical_harmonics.kernels import tf_eval_monom_basis, tf_monom_basis_offset, tf_monomial_basis_3D_idx, compute_monomial_basis_offset
from spherical_harmonics.kernels import real_spherical_harmonic, zernike_kernel_3D, tf_zernike_kernel_basis
from spherical_harmonics.kernels import A, B, associated_legendre_polynomial
from utils.pointclouds_utils import generate_3d
from network_utils.group_points import GroupPoints
from network_utils.sparse_grid_sampling_eager import GridSampler, GridPooling, extract_batch_idx
from network_utils.pooling import kd_pooling_1d, kd_median_sampling, kdtree_indexing, aligned_kdtree_indexing, kdtree_indexing_, aligned_kdtree_indexing_
from utils.pointclouds_utils import np_kd_tree_idx, pc_batch_preprocess
from data_providers.classifiaction_provider import ClassificationProvider
from data_providers.classification_datasets import datsets_list
from spherical_harmonics.clebsch_gordan_decomposition import tf_clebsch_gordan_decomposition
from network_utils.pooling import extract_samples_slices
from time import time
from sklearn.neighbors import NearestNeighbors
import h5py
from sympy import *
x, y, z = symbols("x y z")


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

print(tf.__version__)
import numpy as np
from utils.data_prep_utils import load_h5_data
from pooling import GridBatchSampler
from utils.pointclouds_utils import setup_pcl_viewer
import vispy
from activations import tf_dodecahedron_sph
from spherical_harmonics.kernels import monomial_basis_3D
from pooling import simple_grid_sampler


def norm2(x, keepdims=False):
  y = tf.sqrt(tf.reduce_sum(tf.multiply(x, x), axis=-1, keepdims=keepdims))
  return y

def shapes(x):
    if isinstance(x, dict):
        y = dict()
        for k in x:
            if isinstance(x[k], list):
                L = []
                for xij in x[k]:
                    L.append(xij.shape)
                y[k] = L
            else:
                y[k] = x[k].shape
    if isinstance(x, list):
        y = []
        for xi in x:
            if isinstance(xi, list):
                L = []
                for xij in xi:
                    L.append(xij.shape)
                y.append(L)
            else:
                y.append(xi.shape)
    return y

"""
a = [1, 2, 3]
print(a.__contains__(10))


x = tf.constant([[2, 6], [3, 4], [1, 2]])
y = tf.constant([0, 1, 2, 3])
print(x.shape)
print(tf.pow(x, y))
print(tf.range(3))
"""
"""
a = (1, 2, 3)
print(a[2])


x = tf.constant([[2, 6], [3, 4], [1, 2]])

y = tf.constant([True, False, True])
mask = x < 3
print(mask)
"""
"""
keys_tensor = tf.constant([1, 2])
vals_tensor = tf.constant([3, 4])
input_tensor = tf.constant([1, 5, 2])
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
print(table.lookup(input_tensor))

x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])
x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])

print(tf.unique_with_counts(x)[2])
print(tf.pad(tf.unique_with_counts(x)[2], [[1, 0]]))
print(tf.cumsum( tf.pad(tf.unique_with_counts(x)[2], [[1, 0]]) )[:-1])
print(np.unique(x, return_index=1)[1])

def unique_with_inverse(x):
    y, idx = tf.unique(x)
    num_segments = tf.shape(y)[0]
    num_elems = tf.shape(x)[0]
    return (y, idx,  tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments))

print(unique_with_inverse(x)[-1])
"""

"""
g = GridSampler(1/16., 1024)
# g_ = GridSampler_(1/8., 1024)
X = load_h5_data("E:/Users/Adrien/Documents/Datasets/dfaust_pointclouds_1024/test_0.hdf5")
y = g(X)[0]
# y_ = g_(X)[0]
print('zzzzzzzz')
# print(tf.reduce_sum(tf.multiply(y_-y, y_-y)))

print(X.shape)
print(y[0].shape)
z = np.array(y[0])
# print(z)
# print(y[-1][0])
setup_pcl_viewer(X=z, color=(1, 1, 1, .5), run=True)
vispy.app.run()
"""

"""
u = tf.range(-1, 2)
x, y, z = tf.meshgrid(u, u, u)
print(u)
g = tf.stack(tf.meshgrid(u, u, u), axis=-1)
print(g.shape)
"""

"""
keys_tensor = tf.constant([1, 2])
vals_tensor = tf.constant([3, 4])
input_tensor = tf.constant([1, 5])
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
print(table.lookup(input_tensor))
"""
"""
def tf_eval_monoms_(x, d, idx=None):
    batch_size = x.shape[0]
    num_points = x.shape[1]
    if idx is None:
        idx = tf_monomial_basis_3D_idx(d)
    y = []
    for i in range(3):
        pows = tf.reshape(tf.range(d+1, dtype=tf.float32), (1, d+1))
        yi = tf.pow(tf.expand_dims(x[..., i], axis=-1), pows)
        y.append(tf.gather(yi, idx[..., i], axis=-1))
    y = tf.stack(y, axis=-1)
    y = tf.reduce_prod(y, axis=-1, keepdims=False)
    return y

x = tf.random.uniform((1024, 3))
x = tf.nn.l2_normalize(x, axis=-1)
# x = tf.multiply(x, tf.constant([[1., 0.33, 1]]))
y = tf.constant([[1., 1., 1.]])

monoms_x = tf_eval_monoms_(x, d=3)

l = 1
Z = tf_zernike_kernel(3, l, l)

R = tf.convert_to_tensor(generate_3d(), dtype=tf.float32)
print(tf.einsum('ij,kj->ik', R, R))
monoms_Rx = tf_eval_monoms_(tf.einsum('ij,pj->pi', R, x), d=3)
print(monoms_x.shape)
zm = tf.einsum('ij,pj->pi', Z, monoms_x)
zmR = tf.einsum('ij,pj->pi', Z, monoms_Rx)
print(zm.shape)


print(tf.reduce_mean(tf.abs(norm2(zmR) - norm2(zm))))
"""
"""
monoms_y = tf_eval_monoms_(y, d=3)
monoms_z = tf_eval_monoms_(tf.subtract(x, y), d=3)
coeffs, idx = tf_monom_basis_offset(d=3)

print(idx.shape)
print(monoms_y.shape)
offset_matrix = tf.gather(monoms_y, idx, axis=-1)
offset_matrix = tf.multiply(offset_matrix, coeffs)
print(offset_matrix.shape)

monoms_x_ = tf.einsum('ij,...j->...i', offset_matrix[0, ...], monoms_x)

print(monoms_x_ - monoms_z)
"""

"""
def real_spherical_harmonic_(l, m, x, y, z):
  K = np.sqrt((2 * l + 1) / (2 * np.pi))
  r2 = x ** 2 + y ** 2 + z ** 2
  if m > 0:
    Ylm = K * associated_legendre_polynomial(l, m, z, r2) * A(m, x, y)
  elif m < 0:
    Ylm = K * associated_legendre_polynomial(l, -m, z, r2) * B(-m, x, y)
  else:
    K = np.sqrt((2 * l + 1) / (4 * np.pi))
    Ylm = K * associated_legendre_polynomial(l, 0, z, r2)
  return Ylm

for l in range(3+1):
  print("     " + str(l))
  for m in range(2*l+1):
    print(m-l)
    print(poly(real_spherical_harmonic_(l, m-l, x, y, z), x, y, z))

"""

radius = 0.2
group_pts = GroupPoints(radius="10", patch_size_source=64)




#  X = load_h5_data("E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_original/data_hdf5/test_data_0.h5")



dataset = datsets_list[0]

provider = ClassificationProvider(files_list=dataset['train_files_list'],
                                  data_path=dataset['train_data_folder'],
                                  n_classes=dataset['num_classes'],
                                  n_points=2048,
                                  batch_size=64,
                                  preprocess=dataset['train_preprocessing'],
                                  shuffle=False,
                                  classes=dataset['classes'])






num_points_input = 2048
num_points = [1024, 512, 128, 32]
# num_points = [num_points[0]] + num_points
cells_size = [0.05, 0.1, 0.2, 0.4]
patch_size = [64, 64, 64]
spacing = [0, 0, 0, 0]
radius_ = ["10", "10", "10"]
radius_ = ["avg", "avg", "avg"]
patches_r = [0.]*(len(num_points)-1)
shape_r = [0.]*(len(num_points)-1)
shapes = [0]*(len(num_points))
j = 0
for x, _ in provider:
    y = tf.convert_to_tensor(x, dtype=tf.float32)
    p = []
    # shapes[0] = x.shape[1]
    for i in range(len(num_points)):
        pool_size = int(num_points_input/num_points[i])
        pi = kd_pooling_1d(x, pool_size=pool_size)
        # print(x.shape)
        # pi = simple_grid_sampler(x, cell_size=cells_size[i], pool_size=pool_size)
        # print('shape', i, pi.shape)
        shapes[i] = pi.shape[1]
        y = pi
        # print("pi shape ", pi.shape)
        p.append(pi)
    g = []

    for i in range(len(p)-1):
        gi = GroupPoints(radius=radius_[i], patch_size_source=patch_size[i],
                         spacing_source=spacing[i])({"source points": p[i], "target points": p[i+1]})
        # print(gi)
        patches_r[i] += tf.reduce_mean(gi["patches radius source"])
        g.append(gi)
        shape_radius = norm2(p[i])
        shape_radius = tf.reduce_max(shape_radius, axis=-1, keepdims=False)
        shape_radius = tf.reduce_mean(shape_radius, keepdims=False)
        shape_r[i] += shape_radius
    j += 1.

print("shapes", shapes)
for i in range(len(p)-1):
    print(i)
    print('shape r ',  shape_r[i] / j)
    print('patches r', patches_r[i] / j)

"""
Q = tf_clebsch_gordan_decomposition(l_max=3, l_max_out=2)

T = [tf.ones((1, 1, 3, 3, 1)), tf.ones((1, 1, 5, 5, 1))]

print(shapes(Q.decompose(T)))
"""
"""
P = tf_pentakis_dodecahedron_sph(3)
for l in P:
    print(tf.reduce_sum(tf.multiply(P[l], P[l])))
"""

"""
# print(tf.range(-2, 3, dtype=tf.float32))

print(len(monomial_basis_3D(3)))

M = tf.constant([[2., 0.], [0., 3.], [0., 4.]])
print(M.shape)
v = tf.ones((2, 2, 2, 3))
x = tf.matmul(M, v)
print(x)
print(x.shape)

idx = tf.constant([0, 0, 0, 1, 1, 2, 3, 3, 3], dtype=tf.int32)
vals = tf.constant([1, 2, 3, 1, 2, 1, 1, 2, 3], dtype=tf.float32)
print(tf.math.unsorted_segment_max(vals, idx, 5))
"""

"""
X = load_h5_data("E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_2048_original/data_hdf5/test_data_0.h5")

batch_size = 32
num_points = 1024
dataset = datsets_list[0]
provider = ClassificationProvider(files_list=dataset['train_files_list'],
                                  data_path=dataset['train_data_folder'],
                                  n_classes=dataset['num_classes'],
                                  n_points=num_points,
                                  batch_size=batch_size,
                                  preprocess=dataset['train_preprocessing'],
                                  shuffle=False,
                                  classes=dataset['classes'])

idx = 7 # plane
idx = 28
idx = 21
x, _ = provider.__getitem__(1)
x = tf.convert_to_tensor(x, dtype=tf.float32)
batch_idx = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1)
batch_idx = tf.tile(batch_idx, (1, num_points))
batch_idx = tf.reshape(batch_idx, (-1))
x = tf.reshape(x, (-1, 3))
x_orig = x
x = {"batch idx": batch_idx, "points": x}
x = GridSampler(cell_size=0.05, limit=1200)(x)

y = GridPooling('avg')([x_orig, x["cell idx"], x["num cells"]])
print(x["rounded points"].shape)
print(y.shape)


# x = {"batch idx": x["batch idx"], "points": x["rounded points"]}
# x = GridSampler(cell_size=0.05, limit=2560)(x)

y = extract_batch_idx(i=idx, x=y, batch_idx=x["batch idx"])
x = extract_batch_idx(i=idx, x=x["rounded points"], batch_idx=x["batch idx"])

print(x.shape)
print(y.shape)
z = np.array(y)

# print(z)
# print(y[-1][0])
setup_pcl_viewer(X=z, color=(1, 1, 1, .5), run=True)
vispy.app.run()
"""

"""
batch_idx = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1)
batch_idx = tf.tile(batch_idx, (1, num_points))
batch_idx_ = tf.reshape(batch_idx, (-1))
k = 0
scale = 0.07
scales = [scale, scale*2., scale*4]
limit = [1024, 256, 64]
limit = [None, None, None]
# scales = [scale, scale*4., scale*16]





start_time = time()
num_cells_max = [0]*len(scales)
num_cells_avg = [0]*len(scales)
num_cells_min = [10000000000]*len(scales)
for x, _ in provider:
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.reshape(x, (-1, 3))
    batch_idx = batch_idx_
    for i in range(len(scales)):
        # points = tf.tile(x, (1, 40))
        x = {"batch idx": batch_idx, "points": x}
        y = GridSampler(cell_size=scales[i], limit=limit[i])(x)
        # GridPooling('avg')([points, y["cell idx"], y["num cells"]])
        x = y["rounded points"]
        batch_idx = y["batch idx"]
        num_cells_max[i] = max(num_cells_max[i], y["num cells"])
        num_cells_min[i] = min(num_cells_min[i], y["num cells"])
        num_cells_avg[i] += y["num cells"]
    k += batch_size

time_ = (time()-start_time)
print(time_)

for i in range(len(scales)):
    print('num cells min ', num_cells_min[i] / batch_size)
    print('num cells avg ', num_cells_avg[i] / k)
    print('num cells max ', num_cells_max[i] / batch_size)
"""
"""
vals = tf.constant([1, 1, 1, 1, 1, 4, 1, 1, 1])
idx = tf.constant([-1, 2, 0, 1, 1, 4, -1, -1, -1])

print(tf.math.unsorted_segment_sum(vals, idx, 5))
"""

""""
X = load_h5_data("E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_fps_multiscale/data_hdf5/test_data_0.h5")
num_points = [1024, 256, 64, 16]
k = extract_samples_slices(X.shape[1], num_points)
print("k ", k)
i = 2
x = X[0, k[i]-num_points[i]:k[i], :]

setup_pcl_viewer(X=x, color=(1, 1, 1, .5), run=True)
vispy.app.run()
"""

"""
X = np.random.rand(32, 1043, 3)
Y = np.random.rand(32, 512, 3)
neigh = NearestNeighbors(n_neighbors=32)
neigh.fit(X)
dist, idx = neigh.kneighbors(Y)
print(idx.shape)
"""

X = load_h5_data("E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_2048_original/data_hdf5/test_data_0.h5")

x = X[16, ...]
x = np.expand_dims(x, axis=0)
# x = pc_batch_preprocess(x, f=None, y=None, proc='rotate')
# x = pc_batch_preprocess(x, f=None, y=None, proc='kd_tree_idx')
x, idx, inv_idx = aligned_kdtree_indexing_(x)

# x = tf.gather_nd(x, idx)

# x = kdtree_indexing(x)
# 64 0.2
# 256 0.1
num_points = [2048, 1024, 512, 256, 64, 32, 16]
num_points = [num_points[0]] + num_points
cells_size = [0.05, 0.1, 0.2, 0.4]
points = []
xi = x
for i in range(len(num_points)-1):
    # pool_size = int(num_points[i] / num_points[i+1])
    # xi = simple_grid_sampler(xi, cell_size=cells_size[i], pool_size=pool_size)
    pool_size = int(num_points[0] / num_points[i+1])
    # xi = simple_grid_sampler(x, cell_size=cells_size[i], pool_size=pool_size)
    xi = kd_pooling_1d(x, pool_size=pool_size)
    # xi = kd_median_sampling(x, pool_size)
    print(xi.shape)

    points.append(xi)

x = points[3][0, ...]

# x = kd_pooling_1d(x, pool_size=4)
# x = x[0, ...]
x = np.array(x)

print(x.shape)
setup_pcl_viewer(X=x, color=(1, 1, 1, .5), run=True)
vispy.app.run()

"""
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

h5_filename = "E:/Users/Adrien/Documents/Datasets/ScanObjectNN_h5_files/main_split_nobg/test_objectdataset.h5"
# f = h5py.File(h5_filename, mode='r')


data, labels = load_h5(h5_filename)

print(data.shape)
print(labels.shape)


x = data[18, ...]
x = np.array(x)

print(x.shape)
setup_pcl_viewer(X=x, color=(1, 1, 1, .5), run=True)
vispy.app.run()
"""




