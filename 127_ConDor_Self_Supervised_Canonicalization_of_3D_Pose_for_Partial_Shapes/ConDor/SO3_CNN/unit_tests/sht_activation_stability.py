import tensorflow as tf
import numpy as np
from SO3_CNN.sampling import tf_S2_fps
from SO3_CNN.spherical_harmonics_ import SphericalHarmonicsEval, SphericalHarmonicsCoeffs, tf_spherical_harmonics

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

def tf_random_rotate(x):
    R = tf_random_rotation(x.shape[0])
    return tf.einsum('bij,bpj->bpi', R, x)


class SphericalHarmonicsEval_:
    def __init__(self, base, l_max=3, l_list=None, sph_fn=None):
        self.base = base
        if sph_fn is not None:
            if l_list is not None:
                self.l_list = l_list
                self.l_max = max(l_list)
            else:
                self.l_list = range(l_max+1)
                self.l_max = l_max
            self.sph_fn = sph_fn
        else:
            self.sph_fn = tf_spherical_harmonics(l_max=l_max, l_list=l_list)


        S2 = base
        y = self.sph_fn.compute(S2)
        self.types = y.keys()
        Y = []
        for l in self.types:
            # Y.append(tf.reshape(y[l], (S2.shape[0], -1, 2*int(l)+1)))
            Y.append(y[l])
        self.Y = tf.concat(Y, axis=-1)

    def compute(self, x):
        """
        X = []

        for l in self.types:
            X.append(x[l])
        X = tf.concat(X, axis=-2)
        """
        return tf.einsum('bvm,nm->bvn', self.Y, x)


class SphericalHarmonicsCoeffs_:
    def __init__(self, base, l_max=3, l_list=None, sph_fn=None):
        self.base = base
        if l_list is not None:
            self.l_list = l_list
            self.l_max = max(l_list)
        else:
            self.l_list = list(range(l_max + 1))
            self.l_max = l_max

        self.split_size = []
        for i in range(len(self.l_list)):
            self.split_size.append(2*self.l_list[i] + 1)

        if sph_fn is not None:
            self.sph_fn = sph_fn
        else:
            self.sph_fn = tf_spherical_harmonics(l_max=l_max, l_list=l_list)


        S2 = self.base



        y = self.sph_fn.compute(S2)

        self.types = list(y.keys())
        Y = []
        for l in self.types:
            # Y.append(tf.reshape(y[l], (S2.shape[0], -1, 2*int(l)+1)))
            Y.append(y[l])
        self.Y = tf.concat(Y, axis=-1)
        self.S2 = S2

    def compute(self, x):
        X = []


        c = tf.einsum('bvm,bvn->bmn', self.Y, x) / (self.Y.shape[1] / (4*np.pi))

        """
        c = tf.split(c, num_or_size_splits=self.split_size, axis=-2)

        C = dict()
        for i in range(len(self.types)):
            l = self.types[i]
            sl = list(x.shape)
            sl[-2] = 2*int(l)+1
            C[l] = tf.reshape(c[i], sl)
        """
        return c

    def get_samples(self):
        return self.S2

def tf_fibonnacci_sphere_sampling(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    S2 = np.stack([x, y, z], axis=-1)
    return tf.convert_to_tensor(S2, dtype=tf.float32)


num_samples = [32]
for i in range(4*46):
    num_samples.append(num_samples[-1] + 1)

batch_size = 10000

l_max = 4









def variance(x):

    c = tf.reduce_mean(x, axis=0, keepdims=True)
    x = tf.subtract(x, c)
    x = tf.multiply(x, x)
    x = tf.reduce_sum(x, axis=-2, keepdims=False)



    v = tf.reduce_mean(x, axis=0, keepdims=False)
    v = tf.sqrt(v)
    v = tf.reduce_mean(v, axis=0, keepdims=False)
    return float(v)

var = dict()
for l in range(l_max+1):
    var[str(l)] = []

for i in range(len(num_samples)):

    # S2 = tf_S2_fps(num_samples=num_samples[i], res=150)
    S2 = tf_fibonnacci_sphere_sampling(num_samples[i])
    samples_batch = tf.expand_dims(S2, axis=0)
    samples_batch = tf.tile(samples_batch, (batch_size, 1, 1))

    rot_batch = tf_random_rotate(samples_batch)

    print(num_samples[i])
    co = SphericalHarmonicsCoeffs_(l_max=l_max, base=rot_batch)
    ev = SphericalHarmonicsEval_(l_max=l_max, base=rot_batch)
    for l in range(1, l_max+1):
        coeffs = np.zeros((2 * l + 1, (l_max + 1) ** 2))
        for i in range(2 * l + 1):
            coeffs[i, i + (l) ** 2] = 1.0

        coeffs = tf.convert_to_tensor(coeffs, dtype=tf.float32)

        y = ev.compute(coeffs)

        y = tf.nn.relu(y)
        y = co.compute(y)

        v = variance(y)

        print(v)
        var[str(l)].append(v)


for l in range(l_max+1):
    print(var[str(l)])

"""
===============================
Legend using pre-defined labels
===============================

Defining legend labels with plots.
"""


import numpy as np
import matplotlib.pyplot as plt

# Make some fake data.
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]



# Create plots with pre-defined labels.
fig, ax = plt.subplots()
colors = ['b', 'g', 'r', 'c']
for l in range(1, l_max+1):

    ax.plot(np.array(num_samples), np.array(var[str(l)]), colors[l-1], label='l = ' + str(l))
    # ax.plot(a, d, 'k:', label='Data length')
    # ax.plot(a, c + d, 'k', label='Total message length')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

"""
import matplotlib
matplotlib.axes.Axes.plot
matplotlib.pyplot.plot
matplotlib.axes.Axes.legend
matplotlib.pyplot.legend
"""
