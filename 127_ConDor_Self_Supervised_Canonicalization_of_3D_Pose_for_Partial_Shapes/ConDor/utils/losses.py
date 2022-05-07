import tensorflow as tf
import numpy as np
from utils.pointclouds_utils import tf_sq_distance_matrix_l2
from utils.data_prep_utils import compute_centroids


def tf_get_keypoint_directions(kps):
    '''
    Function to obtain directions to keypoints
    
    kps - B, N, 3
    returns - B, 3, N, N (directions)
    '''

    num_kps = kps.shape[1]

    kps_temp = tf.transpose(kps, perm=[0, 2, 1]) # B, 3, N
    kps_repeat_1 = tf.repeat(tf.expand_dims(kps_temp, axis = -1), num_kps, axis = -1) # B, 3, N, N
    kps_repeat_2 = tf.repeat(tf.expand_dims(kps_temp, axis = -2), num_kps, axis = -2) # B, 3, N, N
    
    directions = tf.subtract(kps_repeat_1, kps_repeat_2)
        
    return directions


def tf_directional_loss(kps_full, kps_partial, caps_sum_full = None, caps_sum_partial = None):
    '''
    Function to get directional loss weighted by the sum of capsules
    
    kps_full - B, N, 3
    kps_partial - B, N, 3

    caps_sum_full - B, 1, K
    caps_sum_partial - B, 1, K
    '''
    weight_matrix = None
    num_kps = kps_full.shape[1]

    if caps_sum_full is not None:
        eps = 1e-8
        caps_sum_full_max = tf.reduce_sum(caps_sum_full, axis = 2, keepdims = True)
        caps_sum_partial_max = tf.reduce_sum(caps_sum_partial, axis = 2, keepdims = True)
        caps_sum_partial_norm = tf.divide(caps_sum_partial, caps_sum_partial_max + eps)

        caps_sum_full_norm = tf.divide(caps_sum_full, caps_sum_full_max + eps)
        caps_sum_full_temp = tf.transpose(caps_sum_full_norm, perm = [0, 2, 1]) # B, K, 1
        caps_sum_full_repeat = tf.repeat(caps_sum_full_temp, num_kps, axis = -1)
        caps_sum_partial_temp = tf.transpose(caps_sum_partial_norm, perm = [0, 2, 1])
        caps_sum_partial_repeat = tf.repeat(caps_sum_partial_temp, num_kps, axis = -1)
        weight_matrix = tf.keras.activations.tanh(tf.multiply(caps_sum_full_repeat, caps_sum_partial_repeat))


    kps_full_directions = tf_get_keypoint_directions(kps_full)
    kps_partial_directions = tf_get_keypoint_directions(kps_partial)

    loss = tf.keras.losses.CosineSimilarity(axis=1)
    if weight_matrix is not None:
        # print(weight_matrix.shape) # B, K, K
        # similarity_loss =  1 + tf.multiply(tf.keras.losses.cosine_similarity(kps_full_directions, kps_partial_directions, axis = 1), weight_matrix)
        similarity_loss =  1 + tf.multiply(loss(kps_full_directions, kps_partial_directions), weight_matrix)
    else:
        # similarity_loss =  1 + tf.keras.losses.cosine_similarity(kps_full_directions, kps_partial_directions, axis = 1)
        similarity_loss =  1 + loss(kps_full_directions, kps_partial_directions)


    return tf.reduce_mean(similarity_loss)


def localization_loss(points, capsules, centroids=None):
    if centroids is None:
        centroids = compute_centroids(points, capsules)
    D2 = sq_distance_mat(points, centroids)
    l = tf.einsum('bic,bic->bc', capsules, D2)
    return tf.reduce_mean(l)

def localization_loss_new(points, capsules, centroids):


    points_centered = points[:, :, None] - centroids[:, None, :] # B, N, K, 3
    points_centered_activated = capsules[:, :, :, None] * points_centered

    l = tf.transpose(points_centered, perm=[0, 2, 1, 3]) # B, K, N, 3
    l_1 = tf.transpose(points_centered_activated, perm=[0, 2, 3, 1]) # B, K, 3, N

    covariance = l_1 @ l
    loss = tf.reduce_mean(tf.linalg.diag_part(covariance))
    return loss

def equilibrium_loss(unnormalized_capsules):
    a = tf.reduce_mean(unnormalized_capsules, axis=1, keepdims=False)
    am = tf.reduce_mean(a, axis=-1, keepdims=True)
    l = tf.subtract(a, am)
    l = l*l
    return tf.reduce_mean(l)


def repulsive_loss_2(points, capsules, sq_distance_mat_=None):
    if sq_distance_mat_ is None:
        D2 = sq_distance_mat(points, points)
    else:
        D2 = sq_distance_mat_
    l = tf.einsum('bij,bjc->bic', D2, capsules)
    l = tf.einsum('bic,bid->bcd', l, capsules)
    return tf.reduce_mean(l)


def l2_loss_(x, y):
    z = x - y
    z = z*z
    z = tf.reduce_sum(z, axis=-1)
    z = tf.sqrt(z + 0.0000001)
    return tf.reduce_mean(z)


def orthogonality_loss(R):
    batch_size = R.shape[0]
    RR = tf.matmul(R, R, transpose_a=True)
    RR_ = tf.matmul(R, R, transpose_b=True)
    eye = tf.expand_dims(tf.eye(3), axis=0)
    eye = tf.tile(eye, (batch_size, 1, 1))
    return 0.5*(tf.reduce_mean(tf.abs(RR - eye)) + tf.reduce_mean(tf.abs(RR_ - eye)))

def sq_distance_mat(X, Y):
    # compute distance mat
    XY = tf.einsum('bic,bjc->bij', X, Y)
    X2 = tf.reduce_sum(tf.multiply(X, X), axis=-1, keepdims=False)
    Y2 = tf.reduce_sum(tf.multiply(Y, Y), axis=-1, keepdims=False)
    X2 = tf.expand_dims(X2, axis=-1)
    Y2 = tf.expand_dims(Y2, axis=-2)
    D2 = tf.add(tf.subtract(X2, 2. * XY), Y2)
    return D2

def hausdorff_distance_l2(X, Y, sq_dist_mat=None):
    if sq_dist_mat is None:
        # compute distance mat
        D2 = sq_distance_mat(X, Y)
    else:
        D2 = sq_dist_mat


    dXY = tf.reduce_max(tf.reduce_min(D2, axis=-1, keepdims=False), axis=1, keepdims=False)
    dXY = tf.sqrt(tf.maximum(dXY, 0.000001))
    dYX = tf.reduce_max(tf.reduce_min(D2, axis=1, keepdims=False), axis=-1, keepdims=False)
    dYX = tf.sqrt(tf.maximum(dYX, 0.000001))
    d = tf.maximum(dXY, dYX)
    return tf.reduce_mean(d)

def hausdorff_distance_l1(X, Y):
    # compute distance mat
    X = tf.expand_dims(X, axis=-1)
    Y = tf.expand_dims(Y, axis=-2)
    D = tf.abs(tf.subtract(X, Y))
    dXY = tf.reduce_max(tf.reduce_min(D, axis=-1, keepdims=False), axis=1, keepdims=False)
    dYX = tf.reduce_max(tf.reduce_min(D, axis=1, keepdims=False), axis=-1, keepdims=False)
    d = tf.maximum(dXY, dYX)
    return tf.reduce_mean(d)

def chamfer_distance_l2(X, Y, sq_dist_mat=None):
    if sq_dist_mat is None:
        # compute distance mat
        D2 = sq_distance_mat(X, Y)
    else:
        D2 = sq_dist_mat
    dXY = tf.sqrt(tf.maximum(tf.reduce_min(D2, axis=-1, keepdims=False), 0.000001))
    dXY = tf.reduce_mean(dXY, axis=1, keepdims=False)
    dYX = tf.sqrt(tf.maximum(tf.reduce_min(D2, axis=1, keepdims=False), 0.000001))
    dYX = tf.reduce_mean(dYX, axis=-1, keepdims=False)
    d = dXY + dYX
    return 0.5*tf.reduce_mean(d)

def proj_distance_l2(X, Y):
    XY = tf.einsum('bic,bjc->bij', X, Y)
    X2 = tf.reduce_sum(tf.multiply(X, X), axis=-1, keepdims=False)
    Y2 = tf.reduce_sum(tf.multiply(Y, Y), axis=-1, keepdims=False)
    X2 = tf.expand_dims(X2, axis=-1)
    Y2 = tf.expand_dims(Y2, axis=-2)
    D2 = tf.add(tf.subtract(X2, 2. * XY), Y2)
    dXY = tf.sqrt(tf.maximum(tf.reduce_min(D2, axis=-1, keepdims=False), 0.000001))
    return tf.reduce_mean(dXY)


def chamfer_distance_l1(X, Y):
    # compute distance mat
    X = tf.expand_dims(X, axis=2)
    Y = tf.expand_dims(Y, axis=1)
    D = tf.abs(tf.subtract(X, Y))
    dXY = tf.reduce_mean(tf.reduce_min(D, axis=-1, keepdims=False), axis=1, keepdims=False)
    dYX = tf.reduce_mean(tf.reduce_min(D, axis=1, keepdims=False), axis=-1, keepdims=False)
    d = (dXY + dYX)
    return 0.5*tf.reduce_mean(d)

def repulsive_regularization_l2(Y, k, h):
    D2 = tf_sq_distance_matrix_l2(Y, Y)
    d2, _ = tf.nn.top_k(-D2, k=k+1)
    d2 = d2[:, :, 1:]
    d2 = -d2
    d = tf.sqrt(tf.add(d2, 0.000001))
    return -tf.reduce_mean(tf.multiply(d, tf.exp(-d2 / (h**2))))

def repulsive_regularization_l2_bis(X, Y, h):
    nb = Y.get_shape()[0]
    nr = X.get_shape()[1]
    Y = tf.reshape(Y, (nb, nr, -1, 3))
    X = tf.expand_dims(X, axis=-2)
    d = tf.subtract(Y, X)
    d2 = tf.multiply(d, d)
    d2 = tf.reduce_sum(d2, axis=-1, keepdims=False)
    d = tf.sqrt(d2 + 0.000001)
    return -tf.reduce_mean(tf.multiply(d, tf.exp(-d2 / (h**2))))

def repulsive_regularization_l2_3(X, Y, h):
    nb = Y.get_shape()[0]
    nr = X.get_shape()[1]
    Y1 = tf.reshape(Y, (nb, nr, 1, -1, 3))
    Y2 = tf.reshape(Y, (nb, nr, -1, 1, 3))
    Y = tf.subtract(Y2, Y1)
    Y = tf.multiply(Y, Y)
    d2 = tf.reduce_sum(Y, axis=-1, keepdims=False)
    d = tf.sqrt(d2 + 0.000001)
    d = tf.reduce_mean(d, axis=[2, 3])
    d2 = tf.multiply(d, d)
    return -tf.reduce_mean(tf.multiply(d, tf.exp(-d2 / (h**2))))

def wgan_loss_(disc_true, disc_gen):
    disc_gen = tf.reduce_mean(disc_gen)
    disc_true = tf.reduce_mean(disc_true)
    return disc_true - disc_gen

def wgan_loss(y_true, y_pred):
    return tf.reduce_mean(y_true*y_pred)

def wgan_gradient_penalty_loss(discriminator, pc_true, pc_gen):
    nb = pc_true.get_shape()[0]
    nv = pc_true.get_shape()[1]
    w = tf.random.uniform((nb, 1, 1))
    d2 = tf_sq_distance_matrix_l2(pc_true, pc_gen)
    idx = tf.argmin(d2, axis=-1, output_type=tf.int32)
    batch_idx = tf.range(nb)
    batch_idx = tf.reshape(batch_idx, (nb, 1))
    batch_idx = tf.tile(batch_idx, (1, nv))
    idx = tf.stack([batch_idx, idx], axis=-1)
    pc_gen = tf.gather_nd(pc_gen, idx)
    averaged_samples = tf.multiply(w, pc_gen) + tf.multiply(1. - w, pc_true)
    y_pred = discriminator(averaged_samples, training=True)
    gradients = tf.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = tf.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = tf.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return tf.reduce_mean(gradient_penalty)


def l2_distance_batch(x, y):
    z = x - y
    z = z * z
    z = tf.reduce_sum(z, axis=-1)
    z = tf.sqrt(z)
    return z

def chamfer_distance_l2_batch(X, Y, sq_dist_mat=None):

    if sq_dist_mat is None:

        # compute distance mat

        D2 = sq_distance_mat(X, Y)

    else:

        D2 = sq_dist_mat

    dXY = tf.sqrt(tf.maximum(tf.reduce_min(

        D2, axis=-1, keepdims=False), 0.000001))

    dXY = tf.reduce_mean(dXY, axis=1, keepdims=False)

    dYX = tf.sqrt(tf.maximum(tf.reduce_min(

        D2, axis=1, keepdims=False), 0.000001))

    dYX = tf.reduce_mean(dYX, axis=-1, keepdims=False)

    d = dXY + dYX

    return 0.5*d

def one_way_chamfer_distance_l2_batch(X, Y, sq_dist_mat=None):

    if sq_dist_mat is None:

        # compute distance mat

        D2 = sq_distance_mat(X, Y)

    else:

        D2 = sq_dist_mat

    dXY = tf.sqrt(tf.maximum(tf.reduce_min(

        D2, axis=-1, keepdims=False), 0.000001))

    dXY = tf.reduce_mean(dXY, axis=1, keepdims=False)

    return dXY
