import tensorflow as tf
from SO3_CNN.sampling import S2_fps, tf_sq_distance_matrix, tf_patches_idx
from SO3_CNN.spherical_harmonics_ import SphericalHarmonicsCoeffs

def sphere_shell_embedding(x, coeffs, r):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    x = tf.subtract(x, c)
    r_ = tf.linalg.norm(x, axis=2, keepdims=True)
    r_ = tf.reduce_max(r_, axis=1, keepdims=True)
    x = tf.divide(x, tf.maximum(r_, 0.0001))

    h = []

    batch_idx_ = tf.expand_dims(tf.range(x.shape[0]), axis=-1)

    for i in range(len(coeffs)):
        s = r[i]*coeffs[i].get_samples()
        sq_dist_mat = tf_sq_distance_matrix(x, s)
        idx = tf_patches_idx(sq_dist_mat, 1)
        batch_idx = tf.tile(batch_idx_, (1, s.shape[0]))
        idx = tf.stack([batch_idx, idx[..., 0]], axis=-1)
        p = tf.gather_nd(x, idx)
        hi = tf.reduce_sum(tf.multiply(p, coeffs[i].get_samples()), axis=-1, keepdims=False)
        hi = tf.expand_dims(hi, axis=-1)
        # hi = coeffs[i].compute(hi)
        h.append(hi)

    h = tf.concat(h, axis=-1)
    h = coeffs[-1].compute(h)
    # for l in h:
    #    h[l] /= (2.*int(l)+1.)
    # h = h[0]
    return h





