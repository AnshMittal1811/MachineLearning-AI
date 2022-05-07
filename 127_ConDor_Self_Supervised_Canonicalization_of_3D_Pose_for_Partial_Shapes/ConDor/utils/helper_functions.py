import tensorflow as tf
import numpy as np

def apply_regularizer(base_model, layer_name, regularizer_type, value):
    regularizer = getattr(tf.keras.regularizers, regularizer_type)(value)
    for layer in base_model.layers:
        # print("sad")
        for attr in [layer_name]:
            if hasattr(layer, attr):
                print("setting regularizer")
                setattr(layer, attr, regularizer)



def slice_idx_data(full_feats_kd, idx_partial_to_full, idx_inv_full, idx_partial):

    '''Slice gathers features from kdtree of full shape and rearranges it to kdtree of partial shape
    full_feats_kd - B x N x K (K can be 3 or num of capsules)
    
    '''

    out = tf.gather_nd(full_feats_kd, idx_inv_full) # kd to input full
    out = tf.gather(out, idx_partial_to_full, batch_dims = 1) # input full to input partial
    out = tf.gather_nd(out, idx_partial) # input partial to kdtree
    
    return out

def step_scheduler(steps, decay_rate, optimizer, epoch):
    '''step lr scheduler for training TFN capsules
    '''

    if epoch in steps:    
        curr_lr = optimizer.lr
        update_lr = curr_lr * decay_rate
        print("Reducing lr from ", str(curr_lr), " to ", str(update_lr))
        optimizer.lr = update_lr
    
    return optimizer

def orthonormalize_basis(basis, pos_det = False):
    ''' 
    Orthonormalize the predicted frames
    '''

    s, u, v = tf.linalg.svd(basis, full_matrices=True)
    orth_basis = tf.stop_gradient(tf.matmul(u, v, transpose_b=True))
    print(orth_basis.shape, "check")
    if pos_det == True:
    
        determinant = tf.linalg.det(orth_basis)    
        #s_new = tf.cond(determinant < tf.constant(0.0), lambda: tf.constant([[1.0, 1.0, -1.0]]), lambda: tf.constant([[1.0, 1.0, 1.0]]))
        _1 = tf.ones_like(determinant)
        s_new = tf.stop_gradient(tf.stack([_1, _1, determinant], axis = -1))
        orth_basis = tf.matmul(u, tf.matmul(tf.linalg.diag(s_new), v, adjoint_b=True))
    print(orth_basis.shape)
    return orth_basis


def normalize_caps(caps, eps = 1e-8):

    caps_sum = tf.reduce_sum(caps, axis = 1, keepdims = True)

    # Normalizing capsules
    normalized_caps = tf.divide(caps, caps_sum + eps)

    return normalized_caps


def compute_l2_loss(x, y):


    l2_loss = x - y
    l2_loss = tf.reduce_sum(tf.multiply(l2_loss, l2_loss), axis=-1, keepdims=False)
    mean_root_square = l2_loss
    l2_loss = tf.sqrt(l2_loss + 1e-8)
    l2_loss = tf.reduce_mean(l2_loss) 

    mean_root_square = tf.reduce_sum(mean_root_square, axis=1, keepdims=False)
    mean_root_square = tf.sqrt(mean_root_square + 1e-8) / x.shape[1]
    mean_root_square = tf.reduce_mean(mean_root_square)

    return l2_loss, mean_root_square

def convert_yzx_to_xyz_basis(basis):

    # basis - N, 3, 3

    rot_y = tf.constant([[np.cos(np.pi / 2), 0, np.sin(np.pi / 2)]
              ,[0,              1,                      0], 
              [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]], basis.dtype)


    rot_z = tf.constant([
                    [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                    [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                    [0, 0, 1]
                    ], dtype = basis.dtype)

    transform = tf.expand_dims(rot_y @ rot_z, axis = 0)
    transform = tf.tile(transform, [basis.shape[0], 1, 1])

    return transform @ basis
