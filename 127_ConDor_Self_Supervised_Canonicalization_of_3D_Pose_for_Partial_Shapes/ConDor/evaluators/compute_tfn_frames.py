import tensorflow as tf
import numpy as np
import h5py, argparse
import os, sys
sys.path.append("../")
from auto_encoder.tfn_capsules_multi_frame import TFN_multi as TFN
import math
from utils.helper_functions import slice_idx_data, orthonormalize_basis, compute_l2_loss, normalize_caps
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

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

def extend_(x, batch_size):
    last_batch = x.shape[0] % batch_size
    if last_batch > 0:
        X_append = []
        for i in range(batch_size - last_batch):
            X_append.append(x[i, ...])
        X_append = tf.stack(X_append, axis=0)
        y = tf.concat([x, X_append], axis=0)
    else:
        y = x
    return y

def var_(x, axis_mean=0, axis_norm=1):
    mean = tf.reduce_mean(x, axis=axis_mean, keepdims=True)
    y = tf.subtract(x, mean)
    yn = tf.reduce_sum(y * y, axis=axis_norm, keepdims=True)
    yn = tf.reduce_mean(yn, axis=axis_mean, keepdims=True)
    return yn, mean

def std_(x, axis_mean=0, axis_norm=1):
    yn, mean = var_(x, axis_mean=axis_mean, axis_norm=axis_norm)
    return tf.sqrt(yn), mean


def pca_align(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    centred_x = tf.subtract(x, c)
    covar_mat = tf.reduce_mean(tf.einsum('bvi,bvj->bvij', centred_x, centred_x), axis=1, keepdims=False)
    _, v = tf.linalg.eigh(covar_mat)

    x = tf.einsum('bij,bvi->bvj', v, centred_x)
    return x

def pca_frame(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    centred_x = tf.subtract(x, c)
    covar_mat = tf.reduce_mean(tf.einsum('bvi,bvj->bvij', centred_x, centred_x), axis=1, keepdims=False)
    _, v = tf.linalg.eigh(covar_mat)
    return tf.linalg.matrix_transpose(v)

def normalize(x, no_var = False):
    s, m = std_(x, axis_mean=1, axis_norm=-1)
    
    if no_var:
        x = tf.subtract(x,m)
    else:
        x = tf.divide(tf.subtract(x, m), s)
    return x

def orth_procrustes(x, y):
    x = normalize(x)
    y = normalize(y)
    xty = tf.einsum('bvi,bvj->bij', y, x)
    s, u, v = tf.linalg.svd(xty)
    r = tf.einsum('bij,bkj->bik', u, v)
    return r

@tf.function
def forward_pass(model, x, frame = False):

    caps, inv, basis = model(x, training = False)
    
    if not frame:
        print("not using frame")
        return caps, inv, basis

    basis = tf.stack(basis, axis = 1)
    orth_basis = orthonormalize_basis(basis)
    
    inv_combined = tf.einsum('bvij,bkj->bvki', orth_basis, inv)
    inv_combined = tf.stack([inv_combined[..., 2], inv_combined[..., 0], inv_combined[..., 1]], axis=-1)
    
    error_full = tf.reduce_mean(tf.reduce_mean(tf.abs(x[:, None] - inv_combined), axis = -1), axis = -1)
    values, indices = tf.math.top_k(-error_full, k = 1)
    
    orth_basis = tf.squeeze(tf.gather(orth_basis, indices, batch_dims = 1), axis = 1)
    print("using network basis")
    
    rot_y = tf.constant([[np.cos(np.pi / 2), 0, np.sin(np.pi / 2)]
              ,[0,              1,                      0], 
              [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]])


    rot_z = tf.constant([
                    [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                    [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                    [0, 0, 1]
                    ])

    transform = tf.expand_dims(rot_y @ rot_z, axis = 0)
    transform = tf.tile(transform, [orth_basis.shape[0], 1, 1])
    orth_basis = transform @ orth_basis
    
    return caps, inv, orth_basis

def orth_procrustes_new(x, y):
    x = normalize(x, no_var = True)
    y = normalize(y, no_var = True)
    xty = tf.einsum('bvi,bvj->bij', y, x)
    # xty = tf.einsum('bvi,bvj->bij', x, y)
    s, u, v = tf.linalg.svd(xty)
    r = tf.einsum('bij,bkj->bik', u, v)

    dets = tf.linalg.det(r)

    r_new = []
    for i in range(dets.shape[0]):

        if dets[i] < 0:
            u_ins = u[i]
            v_ins = tf.stack([v[i, :, 0], v[i, :, 1], -v[i, :, 2]], axis = -1)
            s_ins = s[i]
            r_ins = tf.einsum('ij,kj->ik', u_ins, v_ins)
        else:
            r_ins = r[i]

        r_new.append(r_ins)
    
    r = tf.stack(r_new, axis = 0)

    return r

def diameter(x, axis=-2, keepdims=True):
    return tf.reduce_max(x, axis=axis, keepdims=keepdims) - tf.reduce_min(x, axis=axis, keepdims=keepdims)

def Log2(x):
    return (math.log10(x) / math.log10(2))
def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) == math.floor(Log2(n)))

def kdtree_indexing(x, depth=None):
    num_points = x.shape[1]
    assert isPowerOfTwo(num_points)
    if depth is None:
        depth = int(np.log(num_points) / np.log(2.) + 0.1)
    y = x
    batch_idx = tf.range(x.shape[0],dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, (-1, 1))
    batch_idx = tf.tile(batch_idx, (1, x.shape[1]))

    for i in range(depth):
        y_shape = list(y.shape)
        diam = diameter(y)
        split_idx = tf.argmax(diam, axis=-1, output_type=tf.int32)
        split_idx = tf.tile(split_idx, (1, y.shape[1]))
        # split_idx = tf.tile(split_idx, (1, y.shape[1], 1))
        idx = tf.range(y.shape[0])
        idx = tf.expand_dims(idx, axis=-1)
        idx = tf.tile(idx, (1, y.shape[1]))
        branch_idx = tf.range(y.shape[1])
        branch_idx = tf.expand_dims(branch_idx, axis=0)
        branch_idx = tf.tile(branch_idx, (y.shape[0], 1))
        split_idx = tf.stack([idx, branch_idx, split_idx], axis=-1)
        m = tf.gather_nd(y, split_idx)
        sort_idx = tf.argsort(m, axis=-1)
        sort_idx = tf.stack([idx, sort_idx], axis=-1)
        y = tf.gather_nd(y, sort_idx)
        y = tf.reshape(y, (-1, int(y.shape[1] // 2), 3))

    y = tf.reshape(y, x.shape)
    return y

def save_pca_frames(filename, shapes_src_path, rot_src_path, tar_path, batch_size=32, num_rots=128):
    fx = h5py.File(os.path.join(shapes_src_path, filename), 'r')
    x = fx['data'][:]
    fx.close()
    num_shapes = x.shape[0]
    fr = h5py.File(os.path.join(rot_src_path))
    r = fr['data'][:]
    print(r.shape)
    print(x.shape)



    fr.close()
    x = extend_(x, batch_size)

    num_batches = x.shape[0] // batch_size
    R = []
    for i in range(num_batches):
        print(100.*i/num_batches)
        xi = x[i * batch_size:(i + 1) * batch_size, ...]
        Ri = []
        for j in range(num_rots):
            rj = r[j, ...]
            xij = tf.einsum("bij,bvj->bvi", rj, xi)
            # yij = pca_frame(xij)
            Ri.append(pca_frame(xij))
        Ri = tf.stack(Ri, axis=1)
        R.append(np.asarray(Ri, dtype=np.float))
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]

    h5_fout = h5py.File(os.path.join(tar_path, filename), 'w')
    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()

def save_pca_frames_(filename, shapes_src_path, rot_src_path, tar_path, batch_size=32, num_rots=128):
    fx = h5py.File(os.path.join(shapes_src_path, filename), 'r')
    x = fx['data'][:]
    fx.close()
    num_shapes = x.shape[0]
    fr = h5py.File(os.path.join(rot_src_path), 'r')
    r = fr['data'][:]
    print(r.shape)
    print(x.shape)



    fr.close()
    x = extend_(x, batch_size)
    # r = extend_(r, batch_size)

    num_batches = x.shape[0] // batch_size
    R = []
    for i in range(num_batches):
        print(100.*i/num_batches)
        xi = x[i * batch_size:(i + 1) * batch_size, ...]

        Ri = []
        for j in range(num_rots):
            xij = tf.einsum("ij,bvj->bvi", r[j, ...], xi)
            # yij = pca_frame(xij)
            Ri.append(pca_frame(xij))
        Ri = tf.stack(Ri, axis=1)
        R.append(np.asarray(Ri, dtype=np.float))
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]

    h5_fout = h5py.File(os.path.join(tar_path, filename), 'w')
    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()



def save_frames(model, filename, shapes_src_path, rot_src_path, tar_path, batch_size=32, num_rots=128, procrustes = 1):
    fx = h5py.File(os.path.join(shapes_src_path, filename), 'r')
    print(tar_path)
    x = fx['data'][:]
    fx.close()
    num_shapes = x.shape[0]
    fr = h5py.File(os.path.join(rot_src_path))
    r = fr['data'][:]
    print(r.shape)
    print(x.shape)



    fr.close()
    x = extend_(x, batch_size)
    # r = extend_(r, batch_size)

    num_batches = x.shape[0] // batch_size
    R = []
    for i in range(num_batches):
        print(100.*i/num_batches)
        xi = x[i * batch_size:(i + 1) * batch_size, ...]
        # ri = r[i * batch_size:(i + 1) * batch_size, ...]
        Ri = []
        for j in range(num_rots):
            rj = r[j, ...]
            xij = tf.einsum("ij,bvj->bvi", rj, xi)
            xij = kdtree_indexing(xij)
            caps, yij, frame = forward_pass(model, xij, frame = True)
            if procrustes:
                print("using procrustes")
                Ri.append(orth_procrustes(xij, yij))
            else: 
                Ri.append(tf.linalg.inv(frame))
            # Ri.append(orth_procrustes_new(xij, yij))
        
        Ri = tf.stack(Ri, axis=1)
        R.append(np.asarray(Ri, dtype=np.float))
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]

    h5_fout = h5py.File(os.path.join(tar_path, filename), 'w')
    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()



def save_frames__(model, filename, shapes_src_path, rot_src_path, tar_path, batch_size=32, num_rots=128):
    fx = h5py.File(os.path.join(shapes_src_path, filename), 'r')
    x = fx['data'][:]
    fx.close()
    num_shapes = x.shape[0]
    fr = h5py.File(os.path.join(rot_src_path, filename))
    r = fr['data'][:]
    print(r.shape)
    print(x.shape)



    fr.close()
    x = extend_(x, batch_size)
    r = extend_(r, batch_size)

    num_batches = x.shape[0] // batch_size
    R = []
    for i in range(num_batches):
        print(100.*i/num_batches)
        xi = x[i * batch_size:(i + 1) * batch_size, ...]
        ri = r[i * batch_size:(i + 1) * batch_size, ...]
        Ri = []
        for j in range(num_rots):
            rij = ri[:, j, ...]
            xij = tf.einsum("bij,bvj->bvi", rij, xi)
            xij = kdtree_indexing(xij)
            yij = model(xij, training=False)
            Ri.append(orth_procrustes(xij, yij))
        Ri = tf.stack(Ri, axis=1)
        R.append(np.asarray(Ri, dtype=np.float))
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]

    h5_fout = h5py.File(os.path.join(tar_path, filename), 'w')
    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()


def save_frames_(model, filename, shapes_src_path, tar_path, batch_size=32):
    fx = h5py.File(os.path.join(shapes_src_path, filename), 'r')
    x = fx['data'][:]
    fx.close()
    num_shapes = x.shape[0]

    print(x.shape)




    x = extend_(x, batch_size)


    num_batches = x.shape[0] // batch_size
    R = []
    for i in range(num_batches):
        print(100.*i/num_batches)
        xi = x[i * batch_size:(i + 1) * batch_size, ...]
        xi = kdtree_indexing(xi)
        yi = model(xi, training=False)
        R.append(np.asarray(orth_procrustes(xi, yi), dtype=np.float))
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]
    R = np.expand_dims(R, axis=1)
    h5_fout = h5py.File(os.path.join(tar_path, filename), 'w')
    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()







def build_model(model_path, model_name, batch_size=32, num_points=1024):
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, num_points, 3))
    alignnet = tf.keras.models.Model(inputs=inputs, outputs=TFN(num_frames=5, num_capsules=10, num_classes=1)(inputs))
    alignnet.load_weights(os.path.join(model_path, model_name))
    model = tf.keras.models.Model(inputs=inputs, outputs=alignnet.outputs[1])
    return model

def build_model_(model_path, model_name, batch_size=32, num_points=1024):
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, num_points, 3))
    alignnet = tf.keras.models.Model(inputs=inputs, outputs=TFN(num_frames=5, num_capsules=10, num_classes=1)(inputs))
    # alignnet.load_weights(os.path.join(model_path, model_name))

    return alignnet

def build_model_new(model_path, batch_size=32, num_points=1024):
    
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, num_points, 3))
    alignnet = tf.keras.models.Model(inputs=inputs, outputs=TFN(num_frames=5, num_capsules=10, num_classes=1)(inputs))
    alignnet.load_weights(model_path)

    return alignnet

# def build_model_partial():

# AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]
# AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "cellphone.h5", "watercraft.h5"]

AtlasNetClasses = ["plane.h5", "chair.h5"]
# AtlasNetClasses = ["bench.h5", "cabinet.h5", "car.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]


AtlasNetShapesPath = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/valid"
AtlasNetRotPath = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_valid"

tfn_full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full"
tfn_full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full_multicategory"
tfn_partial_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_partial"
tfn_partial_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_partial_multicategory"

tfn_full_models_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/models/tfn_full"
tf_partial_models_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/models/tfn_partial"

# multicategory full shapes
"""
for i in range(len(AtlasNetClasses)):
    print(AtlasNetClasses[i])
    model = build_model(tfn_full_models_path, os.path.join(tfn_full_models_path, AtlasNetClasses[i]), batch_size=32, num_points=1024)
    save_frames(model, AtlasNetClasses[i], AtlasNetShapesPath, AtlasNetRotPath, tfn_full_pred_path, batch_size=32, num_rots=128)
"""

pca_full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full"
pca_full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full_multicategory"
pca_partial_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_partial"
pca_partial_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_partial_multicategory"


"""
for i in range(len(AtlasNetClasses)):
    print(AtlasNetClasses[i])
    save_pca_frames(AtlasNetClasses[i], AtlasNetShapesPath, AtlasNetRotPath, pca_partial_multi_pred_path, batch_size=32, num_rots=128)
"""

"""
tfn_full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_consistency_full"
tfn_full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_consistency_full_multicategory"
tfn_partial_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_partial"
tfn_partial_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_partial_multicategory"
"""

def directory_check_create(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description= "Parser for generating frames")

    parser.add_argument("--pca", required = True, type = int)
    parser.add_argument("--num_rots", default = 128, type = int)
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--category", type = str, default = None)
    parser.add_argument("--path_ds", type = str, required = True)
    parser.add_argument("--path_rot", type = str, required = True)
    parser.add_argument("--path_out", type = str, required = True)
    parser.add_argument("--procrustes", type = int, default = 1)

    args = parser.parse_args()
    ########################################################################
    
    model = build_model_new(args.model, batch_size=32, num_points=1024)

    if args.category is not None:
        AtlasNetClasses = [args.category]


    path_out = os.path.join(args.path_out, "")
    directory_check_create(path_out)

    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])

        if args.pca:
            save_pca_frames_(AtlasNetClasses[i], args.path_ds, args.path_rot, path_out, batch_size=32, num_rots=args.num_rots)
        else:
            save_frames(model, AtlasNetClasses[i], args.path_ds, args.path_rot, path_out, batch_size=32, num_rots=args.num_rots, procrustes = args.procrustes)
            








