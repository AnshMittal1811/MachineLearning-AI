import numpy as np
import tensorflow as tf

# compute unnormalized spherical harmonics up to degree l_max <= 3 for a tensor of shape (..., 3)
def unnormalized_complex_sh(l_max, X):
    assert (3 >= l_max > 1)
    Y = list()

    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]

    Y0 = []

    Y00 = tf.constant(value=np.sqrt(1. / np.pi) / 2., shape=x.get_shape(), dtype=tf.complex64)

    Y0.append(Y00)

    Y += Y0

    Y1 = []
    Y1_1 = (np.sqrt(3. / (2. * np.pi)) / 2.) * tf.complex(x, -y)
    Y10 = (np.sqrt(3. / np.pi) / 2.) * tf.complex(z, 0.)
    Y11 = (-np.sqrt(3. / (2. * np.pi)) / 2.) * tf.complex(x, y)

    Y1.append(Y1_1)
    Y1.append(Y10)
    Y1.append(Y11)

    Y += Y1

    if l_max >= 2:
        Y2 = []
        # [x**2, y**2, z**2, x*y, y*z, z*x]
        X2 = tf.multiply(tf.tile(X, (1, 1, 1, 1, 2)), tf.gather(X, [0, 1, 2, 1, 2, 0], axis=-1))

        Y2_2 = (np.sqrt(15. / (2. * np.pi)) / 4.) * tf.complex(X2[..., 0] - X2[..., 1], -2. * X2[..., 3])
        Y2_1 = (np.sqrt(15. / (2. * np.pi)) / 2.) * tf.complex(X2[..., -1], -X2[..., -2])
        Y20 = (np.sqrt(5. / np.pi) / 4.) * tf.complex(2. * X2[..., 2] - X2[..., 0] - X2[..., 1], 0.)
        # Y21 = (-np.sqrt(15./(2.*np.pi))/2.)*tf.complex(X2[..., -1], X2[..., -2])
        # Y22 = (np.sqrt(15./(2.*np.pi))/4.)*tf.complex(X2[..., 0]-X2[..., 1], 2.*X2[..., 3])
        Y21 = -tf.conj(Y2_1)
        Y22 = tf.conj(Y2_2)

        Y2.append(Y2_2)
        Y2.append(Y2_1)
        Y2.append(Y20)
        Y2.append(Y21)
        Y2.append(Y22)

        Y += Y2

    if l_max >= 3:
        # [x**3, y**3, z**3, x**2*y, y**2*z, z**2*x, x**2*z, y**2*x, z**2*y]
        X3 = tf.multiply(tf.tile(X2[..., 0:3], (1, 1, 1, 1, 3)), tf.gather(X, [0, 1, 2, 1, 2, 0, 2, 0, 1], axis=-1))
        xyz = x * y * z

        Y3 = []
        Y3_3 = (np.sqrt(35. / np.pi) / 8.) * tf.complex(X3[..., 0] - 3. * X3[..., -2], X3[..., 1] - 3. * X3[..., 3])
        Y3_2 = (np.sqrt(105. / (2. * np.pi)) / 4.) * tf.complex(X3[..., -3] - X3[..., 4], -2. * xyz)
        Y3_1 = (np.sqrt(21. / np.pi) / 8.) * tf.complex(-X3[..., 0] - X3[..., -2] + 4. * X3[..., -4],
                                                        X3[..., 3] + X3[..., 1] - 4. * X3[..., -1])
        Y30 = (np.sqrt(7. / np.pi) / 4.) * tf.complex(2. * X3[..., 2] - 3. * X3[..., -3] - 3. * X3[..., 4], 0.)
        Y31 = -tf.conj(Y3_1)
        Y32 = tf.conj(Y3_2)
        Y33 = -tf.conj(Y3_3)

        Y3.append(Y3_3)
        Y3.append(Y3_2)
        Y3.append(Y3_1)
        Y3.append(Y30)
        Y3.append(Y31)
        Y3.append(Y32)
        Y3.append(Y33)

        Y += Y3

    # return Y
    return tf.stack(Y, axis=-1)

def normalized_real_sh(l_max, X, r=None, eps=0.001):

    if r is not None:
        X = tf.divide(X, tf.expand_dims(tf.maximum(r, eps), axis=-1))
    else:
        X = tf.nn.l2_normalize(X, axis=-1, epsilon=eps)

    assert (4 >= l_max >= 1)
    Y = list()
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]

    Y0 = []

    Y00 = tf.constant(value=np.sqrt(1. / np.pi) / 2., shape=x.get_shape(), dtype=tf.float32)

    Y0.append(Y00)

    # Y0 = tf.stack(Y0, axis=-1)

    Y.append(Y0)

    Y1 = []
    Y1_1 = (np.sqrt(3. / np.pi) / 2.) * y
    Y10 = (np.sqrt(3. / np.pi) / 2.) * z
    Y11 = (np.sqrt(3. / np.pi) / 2.) * x

    Y1.append(Y1_1)
    Y1.append(Y10)
    Y1.append(Y11)

    Y.append(Y1)

    if l_max >= 2:
        Y2 = []
        # [x**2, y**2, z**2, x*y, y*z, z*x]
        X2 = tf.multiply(tf.tile(X, ( 1, 2)), tf.gather(X, [0, 1, 2, 1, 2, 0], axis=-1))
        x2 = X2[..., 0]
        y2 = X2[..., 1]
        z2 = X2[..., 2]

        Y2_2 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 3]
        Y2_1 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 4]
        Y20 = (np.sqrt(5. / np.pi) / 4.) * (2. * z2 - x2 - y2)
        Y21 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 5]
        Y22 = (np.sqrt(15. / np.pi) / 4.) * (x2 - y2)

        Y2.append(Y2_2)
        Y2.append(Y2_1)
        Y2.append(Y20)
        Y2.append(Y21)
        Y2.append(Y22)

        Y.append(Y2)

    if l_max >= 3:
        # [x**3, y**3, z**3, x**2*y, y**2*z, z**2*x, x**2*z, y**2*x, z**2*y]
        X3 = tf.multiply(tf.tile(X2[..., 0:3], ( 1, 3)), tf.gather(X, [0, 1, 2, 1, 2, 0, 2, 0, 1], axis=-1))
        xyz = x * y * z

        Y3 = []
        Y3_3 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (3. * X3[..., 3] - X3[..., 1])
        Y3_2 = (np.sqrt(105. / np.pi) / 2.) * xyz
        Y3_1 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * X3[..., -1] - X3[..., 3] - X3[..., 1])
        Y30 = (np.sqrt(7. / np.pi) / 4.) * (2. * X3[..., 2] - 3. * X3[..., 6] - 3. * X3[..., 4])
        Y31 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * X3[..., 5] - X3[..., 0] - X3[..., -2])
        Y32 = (np.sqrt(105. / np.pi) / 4.) * (X3[..., -3] - X3[..., 4])
        Y33 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (X3[..., 0] - 3. * X3[..., -2])

        Y3.append(Y3_3)
        Y3.append(Y3_2)
        Y3.append(Y3_1)
        Y3.append(Y30)
        Y3.append(Y31)
        Y3.append(Y32)
        Y3.append(Y33)

        Y.append(Y3)

    if l_max >= 4:
        Y4 = []
        r2 = X2[..., 0] + X2[..., 1] + X2[..., 2]
        r4 = tf.multiply(r2, r2)
        z4 = tf.multiply(X2[..., 2], X2[..., 2])
        x2_y2 = X2[..., 0]-X2[..., 1]
        Y4_4 = (3./4.)*np.sqrt(35./np.pi)*(tf.multiply(X2[..., 3], x2_y2))
        p3x2_y2 = 3.*X2[..., 0] - X2[..., 1]
        Y4_3 = (3./4.)*np.sqrt(35./(2.*np.pi))*(tf.multiply(p3x2_y2, X2[..., -2]))
        p7z2_r2 = 7.*X2[..., 2] - r2
        Y4_2 = (3./4.)*np.sqrt(5./np.pi)*tf.multiply(X2[..., 3], p7z2_r2)
        p7z2_3r2 = 7.*X2[..., 2] - 3.*r2
        Y4_1 = (3./4.)*np.sqrt(5./(2.*np.pi))*tf.multiply(X2[..., -2], p7z2_3r2)
        Y40 = (3./16.)*np.sqrt(1./np.pi)*(35.*z4 - tf.multiply(X2[..., 2], r2) + 3.*r4)
        Y41 = (3./4.)*np.sqrt(5./(2.*np.pi))*(tf.multiply(X2[..., -1], p7z2_3r2))
        Y42 = (3./8.)*np.sqrt(5./np.pi)*tf.multiply(X2[..., 0]-X2[..., 1], p7z2_r2)
        x2_3y2 = X2[..., 0]-3.*X2[..., 1]
        Y43 = (3./4.)*np.sqrt(35./(2.*np.pi))*tf.multiply(x2_3y2, X2[..., -1])
        Y44 = (3./16.)*np.sqrt(35./np.pi)*(tf.multiply(X2[..., 0], x2_3y2) - tf.multiply(X2[..., 1], p3x2_y2))

        Y4.append(Y4_4)
        Y4.append(Y4_3)
        Y4.append(Y4_2)
        Y4.append(Y4_1)
        Y4.append(Y40)
        Y4.append(Y41)
        Y4.append(Y42)
        Y4.append(Y43)
        Y4.append(Y44)

        Y.append(Y4)
    for l in range(len(Y)):
        Y[l] = tf.stack(Y[l], axis=-1)
    return Y


# compute unnormalized spherical harmonics up to degree l_max <= 3 for a tensor of shape (..., 3)
def unnormalized_real_sh(l_max, X):
    assert (4 >= l_max >= 1)
    Y = list()
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]

    Y0 = []

    Y00 = tf.constant(value=np.sqrt(1. / np.pi) / 2., shape=x.get_shape(), dtype=tf.float32)

    Y0.append(Y00)

    # Y0 = tf.stack(Y0, axis=-1)

    Y += Y0

    Y1 = []
    Y1_1 = (np.sqrt(3. / np.pi) / 2.) * y
    Y10 = (np.sqrt(3. / np.pi) / 2.) * z
    Y11 = (np.sqrt(3. / np.pi) / 2.) * x

    Y1.append(Y1_1)
    Y1.append(Y10)
    Y1.append(Y11)

    Y += Y1

    if l_max >= 2:
        Y2 = []
        # [x**2, y**2, z**2, x*y, y*z, z*x]
        X2 = tf.multiply(tf.tile(X, (1, 1, 1, 2)), tf.gather(X, [0, 1, 2, 1, 2, 0], axis=-1))
        x2 = X2[..., 0]
        y2 = X2[..., 1]
        z2 = X2[..., 2]

        Y2_2 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 3]
        Y2_1 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 4]
        Y20 = (np.sqrt(5. / np.pi) / 4.) * (2. * z2 - x2 - y2)
        Y21 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 5]
        Y22 = (np.sqrt(15. / np.pi) / 4.) * (x2 - y2)

        Y2.append(Y2_2)
        Y2.append(Y2_1)
        Y2.append(Y20)
        Y2.append(Y21)
        Y2.append(Y22)

        Y += Y2

    if l_max >= 3:
        # [x**3, y**3, z**3, x**2*y, y**2*z, z**2*x, x**2*z, y**2*x, z**2*y]
        X3 = tf.multiply(tf.tile(X2[..., 0:3], (1, 1, 1, 3)), tf.gather(X, [0, 1, 2, 1, 2, 0, 2, 0, 1], axis=-1))
        xyz = x * y * z

        Y3 = []
        Y3_3 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (3. * X3[..., 3] - X3[..., 1])
        Y3_2 = (np.sqrt(105. / np.pi) / 2.) * xyz
        Y3_1 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * X3[..., -1] - X3[..., 3] - X3[..., 1])
        Y30 = (np.sqrt(7. / np.pi) / 4.) * (2. * X3[..., 2] - 3. * X3[..., 6] - 3. * X3[..., 4])
        Y31 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * X3[..., 5] - X3[..., 0] - X3[..., -2])
        Y32 = (np.sqrt(105. / np.pi) / 4.) * (X3[..., -3] - X3[..., 4])
        Y33 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (X3[..., 0] - 3. * X3[..., -2])

        Y3.append(Y3_3)
        Y3.append(Y3_2)
        Y3.append(Y3_1)
        Y3.append(Y30)
        Y3.append(Y31)
        Y3.append(Y32)
        Y3.append(Y33)

        Y += Y3

    if l_max >= 4:
        Y4 = []
        r2 = X2[..., 0] + X2[..., 1] + X2[..., 2]
        r4 = tf.multiply(r2, r2)
        z4 = tf.multiply(X2[..., 2], X2[..., 2])
        x2_y2 = X2[..., 0]-X2[..., 1]
        Y4_4 = (3./4.)*np.sqrt(35./np.pi)*(tf.multiply(X2[..., 3], x2_y2))
        p3x2_y2 = 3.*X2[..., 0] - X2[..., 1]
        Y4_3 = (3./4.)*np.sqrt(35./(2.*np.pi))*(tf.multiply(p3x2_y2, X2[..., -2]))
        p7z2_r2 = 7.*X2[..., 2] - r2
        Y4_2 = (3./4.)*np.sqrt(5./np.pi)*tf.multiply(X2[..., 3], p7z2_r2)
        p7z2_3r2 = 7.*X2[..., 2] - 3.*r2
        Y4_1 = (3./4.)*np.sqrt(5./(2.*np.pi))*tf.multiply(X2[..., -2], p7z2_3r2)
        Y40 = (3./16.)*np.sqrt(1./np.pi)*(35.*z4 - tf.multiply(X2[..., 2], r2) + 3.*r4)
        Y41 = (3./4.)*np.sqrt(5./(2.*np.pi))*(tf.multiply(X2[..., -1], p7z2_3r2))
        Y42 = (3./8.)*np.sqrt(5./np.pi)*tf.multiply(X2[..., 0]-X2[..., 1], p7z2_r2)
        x2_3y2 = X2[..., 0]-3.*X2[..., 1]
        Y43 = (3./4.)*np.sqrt(35./(2.*np.pi))*tf.multiply(x2_3y2, X2[..., -1])
        Y44 = (3./16.)*np.sqrt(35./np.pi)*(tf.multiply(X2[..., 0], x2_3y2) - tf.multiply(X2[..., 1], p3x2_y2))

        Y4.append(Y4_4)
        Y4.append(Y4_3)
        Y4.append(Y4_2)
        Y4.append(Y4_1)
        Y4.append(Y40)
        Y4.append(Y41)
        Y4.append(Y42)
        Y4.append(Y43)
        Y4.append(Y44)

        Y += Y4

    return tf.stack(Y, axis=-1)


def unnormalized_sh(X, l_max, dtype=tf.float32):
    if dtype is tf.float32:
        return unnormalized_real_sh(l_max, X)
    else:
        return unnormalized_complex_sh(l_max, X)


def normalized_sh(X, l_max, eps, dtype=tf.float32):
    X_ = tf.nn.l2_normalize(X, axis=-1, epsilon=eps)
    return unnormalized_sh(X_, l_max, dtype=dtype)
