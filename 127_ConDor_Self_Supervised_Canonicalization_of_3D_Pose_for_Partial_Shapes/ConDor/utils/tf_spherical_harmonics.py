import numpy as np
import tensorflow as tf
from keras.layers import Layer

def tf_spherical_coordinates(dirs, frames):
    """
    if len(dirs.get_shape().as_list()) != len(frames.get_shape().as_list()):
        dirs = tf.expand_dims(dirs, axis=-1)
    """
    patch_size = dirs.get_shape()[2]

    # frames = tf.expand_dims(frames, axis=2)
    # dirs = tf.expand_dims(dirs, axis=-2)
    # proj = tf.reduce_sum(tf.multiply(frames, dirs), axis=-1, keepdims=False)
    # proj = tf.matmul(frames, dirs, transpose_a=True)
    # proj = tf.einsum('bvij,bvpi->bvpj', frames, dirs)
    proj = dirs
    cp = proj[..., 0]
    sp = proj[..., 1]
    ct = proj[..., 2]
    st = tf.sqrt(tf.maximum(1. - ct*ct, 0.00001))
    # st = tf.sqrt(tf.maximum(cp*cp + sp*sp, 0.00001))
    cp = tf.divide(cp, st)
    sp = tf.divide(sp, st)
    """
    normals = frames[..., 2]
    normals = tf.tile(normals, (1, 1, patch_size))
    st = tf.cross(dirs[..., 0, :], normals)
    st = tf.multiply(st, st)
    st = tf.reduce_sum(st, axis=-1, keepdims=False)
    st = tf.sqrt(tf.maximum(st, 0.0000001))
    """
    return cp, sp, ct, st

def normalize_sh_patch(Y):
    Y = tf.stack(Y, axis=0)
    R = tf.multiply(Y, Y)
    R = tf.reduce_sum(R, axis=0, keepdims=True)
    R = tf.sqrt(tf.maximum(R, 0.0000001))
    R = tf.reduce_mean(R, axis=[1, 2], keepdims=True)
    return tf.divide(Y, R)


"""
def gaussian_shells(r, nr, normalise=False, rad=None):
    if rad is None:
        rad = 1.
        h = 0.75
        sigma_2 = (4.*((nr-1.)**2)*np.log(2.))/(h**2)
    else:
        h = 1.
        sigma = h*rad / (nr-1.)
        sigma_2 = 1./(2.*(sigma**2))

    # sigma_2 = 2*(nr**2)/(h**2)
    # t = h*tf.range(float(nr)) / float(nr)

    t = h*tf.reshape(tf.lin_space(start=0., stop=rad, num=nr), shape=(1, 1, 1, nr))
    d = tf.subtract(tf.expand_dims(r, axis=-1), t)
    g = tf.exp(-sigma_2*tf.multiply(d, d))
    if normalise:
        g_sum = tf.reduce_sum(g, axis=2, keepdims=True)
        g_sum = tf.reduce_mean(g_sum, axis=1, keepdims=True)
        g = tf.divide(g, g_sum)
    g = tf.expand_dims(g, axis=-1)
    return g
"""


def gaussian_shells(d, n):
    assert n >= 1
    x = tf.range(n, dtype=tf.float32) / float(max(n-1, 1))
    k = len(list(d.shape))
    d = tf.expand_dims(d, -1)
    x = tf.reshape(x, [1]*k + [n])
    r = tf.subtract(d, x)
    r2 = tf.multiply(r, r)
    g = tf.exp(4*np.log(0.5)*r2)
    g = tf.divide(g, tf.reduce_sum(g, axis=-1, keepdims=True))
    return g


def unnormalized_real_sh(l_max, X, r):
    X = tf.divide(X, tf.expand_dims(tf.maximum(r, 0.001), axis=-1))
    # g = gaussian_shells(d=r, n=nr)

    # X = tf.divide(X, tf.expand_dims(tf.maximum(r, 0.0001), axis=-1))


    assert (4 >= l_max >= 1)
    Y = list()
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]

    Y0 = []

    Y00 = tf.constant(value=np.sqrt(1. / np.pi) / 2., shape=x.get_shape(), dtype=tf.float32)

    Y0.append(Y00)

    # Y0 = tf.stack(Y0, axis=-1)

    Y.append(tf.stack(Y0, axis=-1))

    Y1 = []
    Y1_1 = (np.sqrt(3. / np.pi) / 2.) * y
    Y10 = (np.sqrt(3. / np.pi) / 2.) * z
    Y11 = (np.sqrt(3. / np.pi) / 2.) * x

    Y1.append(Y1_1)
    Y1.append(Y10)
    Y1.append(Y11)

    Y.append(tf.stack(Y1, axis=-1))

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

        Y.append(tf.stack(Y2, axis=-1))

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

        Y.append(tf.stack(Y3, axis=-1))

    if l_max >= 4:
        Y4 = []
        r2 = X2[..., 0] + X2[..., 1] + X2[..., 2]
        r4 = tf.multiply(r2, r2)
        z4 = tf.multiply(X2[..., 2], X2[..., 2])
        x2_y2 = X2[..., 0] - X2[..., 1]
        Y4_4 = (3. / 4.) * np.sqrt(35. / np.pi) * (tf.multiply(X2[..., 3], x2_y2))
        p3x2_y2 = 3. * X2[..., 0] - X2[..., 1]
        Y4_3 = (3. / 4.) * np.sqrt(35. / (2. * np.pi)) * (tf.multiply(p3x2_y2, X2[..., -2]))
        p7z2_r2 = 7. * X2[..., 2] - r2
        Y4_2 = (3. / 4.) * np.sqrt(5. / np.pi) * tf.multiply(X2[..., 3], p7z2_r2)
        p7z2_3r2 = 7. * X2[..., 2] - 3. * r2
        Y4_1 = (3. / 4.) * np.sqrt(5. / (2. * np.pi)) * tf.multiply(X2[..., -2], p7z2_3r2)
        Y40 = (3. / 16.) * np.sqrt(1. / np.pi) * (35. * z4 - tf.multiply(X2[..., 2], r2) + 3. * r4)
        Y41 = (3. / 4.) * np.sqrt(5. / (2. * np.pi)) * (tf.multiply(X2[..., -1], p7z2_3r2))
        Y42 = (3. / 8.) * np.sqrt(5. / np.pi) * tf.multiply(X2[..., 0] - X2[..., 1], p7z2_r2)
        x2_3y2 = X2[..., 0] - 3. * X2[..., 1]
        Y43 = (3. / 4.) * np.sqrt(35. / (2. * np.pi)) * tf.multiply(x2_3y2, X2[..., -1])
        Y44 = (3. / 16.) * np.sqrt(35. / np.pi) * (tf.multiply(X2[..., 0], x2_3y2) - tf.multiply(X2[..., 1], p3x2_y2))

        Y4.append(Y4_4)
        Y4.append(Y4_3)
        Y4.append(Y4_2)
        Y4.append(Y4_1)
        Y4.append(Y40)
        Y4.append(Y41)
        Y4.append(Y42)
        Y4.append(Y43)
        Y4.append(Y44)

        Y.append(tf.stack(Y4, axis=-1))

    # Y = tf.stack(Y, axis=-1)
    # Y = tf.expand_dims(Y, axis=-2)
    # Y = tf.multiply(g, Y)
    return Y


def unnormalized_real_sh_(l_max, X, r, nr, rad=None):
    batch_size = X.get_shape()[0].value
    num_points = X.get_shape()[1].value
    patch_size = X.get_shape()[2].value

    g = gaussian_shells(r, nr, rad=rad)



    assert (4 >= l_max >= 1)

    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]



    Y00 = tf.constant(value=np.sqrt(1. / np.pi) / 2., shape=x.get_shape(), dtype=tf.float32)

    Y0 = [Y00]

    Y1_11 = (np.sqrt(3. / np.pi) / 2.) * y
    Y100 = (np.sqrt(3. / np.pi) / 2.) * z
    Y1_10 = (np.sqrt(3. / np.pi) / 2.) * x

    Y0.append(Y100)
    Y10 = [Y1_10]
    Y11 = [Y1_11]


    if l_max >= 2:

        # [x**2, y**2, z**2, x*y, y*z, z*x]
        X2 = tf.multiply(tf.tile(X, (1, 1, 1, 2)), tf.gather(X, [0, 1, 2, 1, 2, 0], axis=-1))
        x2 = X2[..., 0]
        y2 = X2[..., 1]
        z2 = X2[..., 2]

        Y2_21 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 3]
        Y2_11 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 4]
        Y200 = (np.sqrt(5. / np.pi) / 4.) * (2. * z2 - x2 - y2)
        Y2_10 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 5]
        Y2_20 = (np.sqrt(15. / np.pi) / 4.) * (x2 - y2)

        Y0.append(Y200)
        Y10.append(Y2_10)
        Y11.append(Y2_11)
        Y20 = [Y2_20]
        Y21 = [Y2_21]

    if l_max >= 3:
        # [x**3, y**3, z**3, x**2*y, y**2*z, z**2*x, x**2*z, y**2*x, z**2*y]
        X3 = tf.multiply(tf.tile(X2[..., 0:3], (1, 1, 1, 3)), tf.gather(X, [0, 1, 2, 1, 2, 0, 2, 0, 1], axis=-1))
        xyz = x * y * z

        Y3_31 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (3. * X3[..., 3] - X3[..., 1])
        Y3_21 = (np.sqrt(105. / np.pi) / 2.) * xyz
        Y3_11 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * X3[..., -1] - X3[..., 3] - X3[..., 1])
        Y300 = (np.sqrt(7. / np.pi) / 4.) * (2. * X3[..., 2] - 3. * X3[..., 6] - 3. * X3[..., 4])
        Y3_10 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * X3[..., 5] - X3[..., 0] - X3[..., -2])
        Y3_20 = (np.sqrt(105. / np.pi) / 4.) * (X3[..., -3] - X3[..., 4])
        Y3_30 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (X3[..., 0] - 3. * X3[..., -2])

        Y0.append(Y300)
        Y10.append(Y3_10)
        Y11.append(Y3_11)
        Y20.append(Y3_20)
        Y21.append(Y3_21)
        Y30 = [Y3_30]
        Y31 = [Y3_31]

    """
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
    """

    if l_max == 1:
        Y_0 = [Y10]
        Y_1 = [Y11]
    elif l_max == 2:
        Y_0 = [Y10, Y20]
        Y_1 = [Y11, Y21]
    else:
        Y_0 = [Y10, Y20, Y30]
        Y_1 = [Y11, Y21, Y31]

    for i in range(len(Y_0)):
        Y_0[i] = tf.stack(Y_0[i], axis=-1)
        Y_1[i] = tf.stack(Y_1[i], axis=-1)

    Y_0 = tf.concat(Y_0, axis=-1)
    Y_1 = tf.concat(Y_1, axis=-1)

    Y = tf.stack([Y_0, Y_1], axis=-1)
    Y = tf.reshape(Y, (batch_size, num_points, patch_size, -1))

    Y0 = tf.stack(Y0, axis=-1)
    Y = tf.concat([Y0, Y], axis=-1)
    Y = tf.expand_dims(Y, axis=-2)
    Y = tf.multiply(g, Y)
    return Y

class SphKernel(Layer):
    def __init__(self, l_max, nr=2, rad=None, **kwargs):
        self.l_max = l_max
        self.nr = nr
        self.rad = rad
        super(SphKernel, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(SphKernel, self).build(input_shape)

    def call(self, x):
        patches = x[0]
        frames = x[1]
        r = x[2]
        batch_size = patches.get_shape()[0].value
        num_points = patches.get_shape()[1].value
        dirs = tf.linalg.l2_normalize(patches, axis=-1)

        print('r ', r.get_shape().as_list())

        """
        if len(r.get_shape().as_list()) == 3:
            dirs = tf.divide(patches, tf.expand_dims(r, axis=-1))
        else:
            dirs = tf.divide(patches, r)
        """

        """
        frames = tf.eye(3)
        frames = tf.reshape(frames, (1, 1, 3, 3))
        frames = tf.tile(frames, (batch_size, num_points, 1, 1))
        """
        proj = tf.einsum('bvij,bvpi->bvpj', frames, dirs)

        # proj = patches
        return unnormalized_real_sh_(self.l_max, proj, r, self.nr, rad=self.rad)



    def compute_output_shape(self, input_shape):
        """
        output_shape = [(input_shape[0][0], input_shape[0][1], input_shape[0][2], self.l_max+1)]
        for i in range(self.l_max):
            output_shape.append((input_shape[0][0], input_shape[0][1], input_shape[0][2], self.l_max-i))
        for i in range(self.l_max):
            output_shape.append((input_shape[0][0], input_shape[0][1], input_shape[0][2], self.l_max-i))
        return output_shape
        """
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.nr, (self.l_max+1)**2)