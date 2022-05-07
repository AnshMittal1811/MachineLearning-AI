import numpy as np
from scipy.spatial import cKDTree, distance_matrix

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
    vertices - point cloud - barycentric coordinates inside triangles

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