import numpy as np
from scipy.spatial import cKDTree, distance_matrix
import os
import h5py
import vispy
import vispy.scene
from vispy.scene import visuals
from functools import partial
from scipy.spatial import cKDTree


# from tqdm import tqdm

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


def multi_res_fps_(x):
    num_points = x.shape[0]
    y = [x]
    while num_points > 1:
        num_points = num_points // 2
        fps_, _ = fps(y[-1], num_points, idx='center')
        y.append(fps_)
    return y


def multi_res_fps(X):
    num_samples = X.shape[0]
    num_points = X.shape[1]
    n = len(multi_res_fps_(X[0, ...]))
    Y = []
    for i in range(n):
        Y.append(np.zeros(shape=(num_samples, num_points, 3)))
        num_points = num_points // 2
    for i in range(num_samples):
        y = multi_res_fps_(X[i, ...])
        for j in range(len(y)):
            Y[j][i, ...] = y[j]
    return Y


def multi_res_fps_idx_(x):
    num_points = x.shape[0]
    y = [np.arange(num_points)]
    while num_points > 1:
        num_points = num_points // 2
        _, idx = fps(x, num_points, idx='center')
        y.append(idx)
    return y


def multi_res_fps_idx(X):
    num_samples = X.shape[0]
    num_points = X.shape[1]
    n = len(multi_res_fps_idx_(X[0, ...]))
    Y = []
    for i in range(n):
        Y.append(np.zeros(shape=(num_samples, num_points), dtype=np.int32))
        num_points = num_points // 2
    for i in range(num_samples):
        print(i / num_samples)
        y = multi_res_fps_idx_(X[i, ...])
        for j in range(len(y)):
            Y[j][i, ...] = y[j]
    return Y


def normalize(X):
    c = np.mean(X, axis=1, keepdims=True)
    X = np.subtract(X, c)
    r = np.multiply(X, X)
    r = np.sum(r, axis=-1, keepdims=True)
    r = np.max(r, axis=1, keepdims=True)
    r = np.maximum(np.sqrt(r), 0.00001)
    X = np.divide(X, r)
    return X


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


def save_h5(h5_filename, data, normals=None, subsamplings_idx=None, part_label=None,
            class_label=None, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)

    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)

    if normals is not None:
        h5_fout.create_dataset(
            'normal', data=normals,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)

    if subsamplings_idx is not None:
        for i in range(len(subsamplings_idx)):
            name = 'sub_idx_' + str(subsamplings_idx[i].shape[1])
            h5_fout.create_dataset(
                name, data=subsamplings_idx[i],
                compression='gzip', compression_opts=1,
                dtype='int32')

    if part_label is not None:
        h5_fout.create_dataset(
            'pid', data=part_label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)

    if class_label is not None:
        h5_fout.create_dataset(
            'label', data=class_label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Read numpy array data and label from h5_filename
def load_h5_data_multires(h5_filename, num_points):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    sub_idx = []
    for i in range(len(num_points)):
        sub_idx.append(f['sub_idx_' + str(num_points[i])])

    class_label = f['class_label'][:]

    if 'part_label' in f:
        part_label = f['part_label'][:]
        return (data, sub_idx, part_label, class_label)
    else:
        return (data, sub_idx, class_label)


def multires_classif(path, file, tar_path):
    data, labels = load_h5(os.path.join(path, file))
    data = normalize(data)
    sub_idx = multi_res_fps_idx(data)
    save_h5(os.path.join(tar_path, file), data, subsamplings_idx=sub_idx, part_label=None,
            class_label=labels, data_dtype='float32', label_dtype='uint8')

def fps_classif(num_points, path, file, tar_path, multiscale=False):
    data, labels = load_h5(os.path.join(path, file))
    # data = normalize(data)
    X = []
    N = []
    num_pts = []
    n = num_points

    if multiscale:
        while n > 0:
            num_pts.append(n)
            n = int(n/2)
    else:
        num_pts = [n]
    num_samples = data.shape[0]
    for i in range(num_samples):
        Xi = []
        y = data[i, ...]
        for j in range(len(num_pts)):
            y, _ = fps(y, num_pts[j], idx=None)
            Xi.append(y)
        Xi = np.concatenate(Xi, axis=0)
        X.append(Xi)
        print(int(100 * i / num_samples))
    X = np.stack(X, axis=0)
    save_h5(os.path.join(tar_path, file), X, subsamplings_idx=None, part_label=None,
            class_label=labels, data_dtype='float32', label_dtype='uint8')

def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename, mode='r')
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return data, label, normal

def fps_classif_normals(num_points, path, file, tar_path, multiscale=False):
    data, labels, normal = load_h5_data_label_normal(os.path.join(path, file))
    # data = normalize(data)
    X = []
    N = []
    num_pts = []
    n = num_points

    if multiscale:
        while n > 0:
            num_pts.append(n)
            n = int(n/2)
    else:
        num_pts = [n]
    num_samples = data.shape[0]
    for i in range(num_samples):
        Xi = []
        Ni = []
        y = data[i, ...]
        n = normal[i, ...]
        for j in range(len(num_pts)):
            y, fps_idx = fps(y, num_pts[j], idx=None)
            n = n[fps_idx, ...]
            Xi.append(y)
            Ni.append(n)
        Xi = np.concatenate(Xi, axis=0)
        Ni = np.concatenate(Ni, axis=0)
        X.append(Xi)
        N.append(Ni)
        print(int(100 * i / num_samples))
    X = np.stack(X, axis=0)
    save_h5(os.path.join(tar_path, file), X, subsamplings_idx=None, part_label=None,
            class_label=labels, data_dtype='float32', label_dtype='uint8', normals=N)

def separate_classes(src_path, files_list, tar_path, file_name, classes):
    data, labels = load_h5_files(src_path, files_list)

    labels = labels[..., 0]

    idx = np.argsort(labels, axis=0)
    labels = labels[idx]
    data = data[idx]
    labels_ = labels.tolist()

    unique_labels = list(set(labels_))
    class_size = []
    for i in range(len(unique_labels)-1):
        class_size.append(labels_.count(unique_labels[i]))
    class_size = np.array(class_size, dtype=np.int32)
    class_size = np.cumsum(class_size)
    data = np.split(data, class_size, axis=0)
    labels = np.split(labels, class_size, axis=0)

    """
    classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
               'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk',
               'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
               'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
               'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    """

    for i in range(len(data)):
        save_h5(os.path.join(tar_path, file_name + '_' + classes[i] + '.h5'), data[i], subsamplings_idx=None,
                part_label=labels[i],
                class_label=None, data_dtype='float32', label_dtype='uint8', normals=None)



def multires_part_seg(path, file, tar_path):
    f = h5py.File(os.path.join(path, file))
    data = normalize(f['data'][:])
    sub_idx = multi_res_fps_idx(data)
    part_labels = f['pid'][:]
    class_labels = f['label'][:]
    save_h5(h5_filename=os.path.join(tar_path, file), data=data, subsamplings_idx=sub_idx,
            part_label=part_labels, class_label=class_labels)


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


def sample_faces(vertices, faces, n_samples=10 ** 4):
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
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    return P, sample_face_idx


def read_and_sample_off(file_path, num_samples=10 ** 4):
    V, F = read_off(file_path)
    return sample_faces(V, F, n_samples=num_samples)


def transfer_labels(x, y, labels):
    T = cKDTree(y)
    _, idx = T.query(x, k=1)

    return labels[idx, ...]

def compute_triangle_normals(V, F):
    xt = np.take(V, F, axis=0)
    u = xt[:, 1, :] - xt[:, 0, :]
    v = xt[:, 2, :] - xt[:, 0, :]
    n = np.cross(u, v)
    n_norm = np.sqrt(np.maximum(np.sum(np.multiply(n, n), axis=-1, keepdims=True), 0.00001))
    n = np.divide(n, n_norm)
    return n

def hdf5_from_off(num_points, tar_path, data_folder, part_labels_folder=None, class_labels_folder=None, n_split=5):
    # list directory
    names_ = []
    for file in os.listdir(data_folder):
        if file.endswith(".off"):
            names_.append(file[:-4])

    # names_ = names_[0:5]

    # split names

    names = []
    num_samples = len(names_)
    b_size = num_samples // n_split
    for i in range(n_split):
        start = i * b_size
        end = min((i + 1) * b_size, num_samples)
        names.append(names_[start:end])

    # print(names[4][23])
    # exit(666)

    original_labels = set([])
    labels_set = set([])
    part_labels = []
    data = []
    normals = []
    avg_num_labels = 0.
    num_shapes = 0.
    for i in range(n_split):
        data.append(np.zeros(shape=(len(names[i]), num_points, 3)))
        normals.append(np.zeros(shape=(len(names[i]), num_points, 3)))
        part_labels.append(np.zeros(shape=(len(names[i]), num_points), dtype=np.int32))
        for j in range(len(names[i])):
            print(i, j)

            V, F = read_off(os.path.join(data_folder, names[i][j] + '.off'))
            nT = compute_triangle_normals(V, F)
            x, face_idx = sample_faces(V, F, n_samples=10000)
            # x = read_and_sample_off(os.path.join(data_folder, names[i][j] + '.off'), num_samples=3*num_points)
            x, fps_idx = fps(x, num_points, idx='center')
            face_idx = face_idx[fps_idx, ...]
            data[i][j, ...] = x
            normals[i][j, ...] = nT[face_idx, ...]

            with open(os.path.join(part_labels_folder, names[i][j] + '.txt'), 'r') as f:
                label = [int(line) for line in f]
                l = set(label)
                original_labels = original_labels.union(l)
                num_shapes += 1.
                avg_num_labels += len(l)
                print(len(l))
                if -1 in label:
                    print(names[i][j])
                labels_set = labels_set.union(l)
            label = np.array(label, np.int32)

            label = transfer_labels(x=x, y=V, labels=label)

            part_labels[i][j, ...] = label

    print('avg_num_labels ', avg_num_labels / num_shapes)

    sub_idx = []

    for i in range(n_split):
        data[i] = normalize(data[i])
        # sub_idx.append(multi_res_fps_idx(data[i]))
        # print('sub_idx shape ', sub_idx[i][0].shape)


    """
    labels_set = set([])
    part_labels = []
    for i in range(len(names)):
        part_labels.append(np.zeros(shape=(len(names[i]), num_points)))
        for j in range(len(names[i])):
            with open(os.path.join(part_labels_folder, names[i][j]+'.txt'), 'r') as f:
                label = [line for line in f]
                labels_set = labels_set.union(set(label))
            label = np.array(label)

            label = transfer_labels(x=data[i][j, ...], )
            part_labels[i][j, ...] = np.array(label, dtype=np.int32)
    """

    labels_set = list(labels_set)

    original_labels = list(original_labels)

    # for i in range(len(original_labels)):

    print('num_labels')
    print(len(labels_set))
    print(labels_set)


    for i in range(n_split):
        for j in range(len(names[i])):
            print(i, j)
            print(part_labels[i].shape)
            for k in range(part_labels[i].shape[1]):

                part_labels[i][j, k] = labels_set.index(part_labels[i][j, k])


    class_labels = []

    for i in range(len(names)):
        class_labels.append(np.zeros(shape=(len(names[i]),), dtype=np.int32))

    for i in range(n_split):
        save_h5(h5_filename=os.path.join(tar_path, 'data_hdf5_' + str(i) + '.h5'),
                data=data[i], normals=normals[i], subsamplings_idx=None,
                part_label=part_labels[i], class_label=None)



def setup_pcl_viewer(X, color=(1, 1, 1, .5), run=False):
    # setup a point cloud viewer using vispy and return a drawing function
    # make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    # create scatter object and fill in the data
    # init_pc = np.random.normal(size=(100, 3), scale=0.2)
    init_pc = X
    scatter = visuals.Markers()
    draw_fn = partial(scatter.set_data, edge_color=None, face_color=color, size=5)
    draw_fn(init_pc)
    view.add(scatter)
    # set camera
    view.camera = 'turntable'  # ['turntable','arcball']
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    return draw_fn

"""
rna_path = 'C:/Users/adrien/Documents/Datasets/RF00001-family'
rna_data_path = os.path.join(rna_path, 'off')
rna_part_labels_path = os.path.join(rna_path, 'labels')
rna_tar_path = os.path.join(rna_path, 'hdf5_data_with_normals_4096')

hdf5_from_off(num_points=4096, tar_path=rna_tar_path, data_folder=rna_data_path,
                  part_labels_folder=rna_part_labels_path,
                  class_labels_folder=None, n_split=5)
"""

modelnet40_rot_path = 'C:/Users/adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_2048_original_rotated'
modelnet40_rot_data_path = os.path.join(modelnet40_rot_path, 'data_hdf5')
modelnet40_rot_tar_path = os.path.join(modelnet40_rot_path, 'data_hdf5_multires')

"""
modelnet40_files = ['test_data_0.h5', 'test_data_1.h5',
                    'train_data_0.h5', 'train_data_1.h5',
                    'train_data_2.h5', 'train_data_3.h5', 'train_data_4.h5']
"""
"""
for i in range(len(modelnet40_files)):
    multires_classif(modelnet40_rot_data_path, modelnet40_files[i], modelnet40_rot_tar_path)
"""
"""
modelnet40_path = 'E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_original'
modelnet40_data_path = os.path.join(modelnet40_path, 'data_hdf5')
# modelnet40_tar_path = os.path.join(modelnet40_path, 'data_hdf5_multires')
# modelnet40_tar_path = "E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_fps_multiscale/data_hdf5"
modelnet40_tar_path = "E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_classes/data_hdf5"


for i in range(len(modelnet40_files)):
    fps_classif_normals(1024, modelnet40_data_path, modelnet40_files[i], modelnet40_tar_path, multiscale=False)

modelnet40_files = ['train_data_0.h5', 'train_data_1.h5',
                    'train_data_2.h5', 'train_data_3.h5', 'train_data_4.h5']

separate_classes(modelnet40_data_path, 'E:/Users/adrien/Documents/Datasets/ModelNet40_hdf5/train_files.txt', modelnet40_tar_path, 'train')
"""

"""
data, sub_idx, class_label = load_h5_data_multires(os.path.join(modelnet40_rot_tar_path, 'test_data_1.h5'),
                                                               [1024, 256, 64])
"""

"""
multires_from_off(num_points=4096, tar_path=rna_tar_path, data_folder=rna_data_path,
                  part_labels_folder=rna_part_labels_path,
                  class_labels_folder=None, n_split=5)
"""
"""
data, sub_idx, part_label, class_label = load_h5_data_multires(os.path.join(rna_tar_path, 'data_hdf5_0.h5'),
                                                               [4096, 1024, 128])
x = data[0, ...]
print(x.shape)
print(sub_idx[2].shape)
x = x[sub_idx[2][0, ...]]
print(x.shape)
# print(sub_idx[1])
setup_pcl_viewer(x)
vispy.app.run()
"""

"""
P = read_and_sample_off(os.path.join(rna_data_path, '1C2X_C.off'), num_samples=10**4)
print('aa')
x, _ = fps(x=P, num_points=2**12)
x = np.expand_dims(x, axis=0)
x = normalize(x)
x = x[0, ...]
print('aa')
y = multi_res_fps_idx_(x)
print(x[y[0]].shape)

setup_pcl_viewer(x[y[0]])
vispy.app.run()
"""

"""
Prepare shape net classifiaction
"""
"""
shapenet_source_path = "E:/Users/Adrien/Documents/Datasets/shapenet_segmentation/hdf5_data"
shapenet_target_path = "E:/Users/Adrien/Documents/Datasets/shapenet_classifiaction"

files_source = ["ply_data_train0.h5",
                "ply_data_train1.h5",
                "ply_data_train2.h5",
                "ply_data_train3.h5",
                "ply_data_train4.h5",
                "ply_data_train5.h5",
                "ply_data_val0.h5",
                "ply_data_test0.h5",
                "ply_data_test1.h5"]

for i in range(len(files_source)):
    fps_classif(1024, shapenet_source_path, files_source[i], shapenet_target_path, multiscale=False)
"""

shapenet_path = 'E:/Users/Adrien/Documents/Datasets/shapenet_classifiaction'

# modelnet40_tar_path = os.path.join(modelnet40_path, 'data_hdf5_multires')
# modelnet40_tar_path = "E:/Users/Adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_1024_fps_multiscale/data_hdf5"
shapenet_tar_path = "E:/Users/Adrien/Documents/Datasets/shapenet_single_class/data_hdf5"


modelnet40_files = ['ply_data_test0.h5',
                    'ply_data_test1.h5',
                    'ply_data_train0.h5',
                    'ply_data_train1.h5',
                    'ply_data_train2.h5',
                    'ply_data_train3.h5',
                    'ply_data_train4.h5',
                    'ply_data_train4.h5',
                    'ply_data_val0.h5']

separate_classes(shapenet_path,
                 'E:/Users/Adrien/Documents/Datasets/shapenet_classifiaction/val_files.txt',
                 shapenet_tar_path,
                 'val', classes=["aero", "bag", "cap", "car", "chair",
                                   "earph", "guitar", "knife", "lamp", "laptop", "motor", "mug",
                                   "pistol", "rocket", "skate", "table"])


