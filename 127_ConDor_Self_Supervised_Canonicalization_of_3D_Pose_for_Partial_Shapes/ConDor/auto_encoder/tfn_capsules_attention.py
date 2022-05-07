import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from spherical_harmonics.kernels import ShGaussianKernelConv
from spherical_harmonics.kernels import ZernikeGaussianKernelConv, ZernikeGaussianKernels, SphericalHarmonicsGaussianKernels
from activations import DodecahedronEval, DodecahedronCoeffs
from group_points import GroupPoints
from pooling import kd_pooling_2d, kd_pooling_1d, kdtree_indexing, aligned_kdtree_indexing
from spherical_harmonics.kernels import tf_monomial_basis_3D_idx, tf_spherical_harmonics_basis, tf_eval_monom_basis
import numpy as np
from utils.pointclouds_utils import tf_kd_tree_idx

from SO3_CNN.spherical_harmonics_ import SphericalHarmonicsCoeffs, SphericalHarmonicsEval
from SO3_CNN.sampling import tf_S2_fps
from SO3_CNN.spherical_harmonics_ import tf_spherical_harmonics


def set_mlp(units, momentum):
    mlp = []
    for i in range(len(units)):
        layer = Dense(units=units[i])
        bn_layer = BatchNormalization(momentum=momentum)
        mlp.append({'layer': layer, 'bn_layer': bn_layer})
    return mlp

def set_mlp_attention(units, momentum, division_factor = 8):
    mlp_attention = []

    for i in range(len(units)):
        query = Dense(units = units[i] // division_factor, name = "query")
        key = Dense(units = units[i] // division_factor, name = "key")
        value = Dense(units = units[i], name = "value")
        # bn_layer = BatchNormalization(momentum=momentum)
        mult_scalar = tf.Variable(1.0, name = "mult")
        add_scalar = tf.Variable(0.0, name = "add")
        mlp_attention.append({'query': query, 'key': key,'value': value, "mult_scalar": mult_scalar, "add_scalar": add_scalar})
    return mlp_attention

def apply_mlp_attention(x, mlp_attention, activation = None):
    """
    Apply attention MLP layers
    """
    y = x
    for i in range(len(mlp_attention)):
        q = mlp_attention[i]["query"](y) # B, N, C//k
        k = mlp_attention[i]["key"](y) # B, N, C//k        
        v = mlp_attention[i]["value"](y)

        # print(q.shape, k.shape, v.shape, y.shape)
        attention = q @ tf.transpose(k, perm = [0, 2, 1]) # B, N, N
        attention = tf.nn.softmax(attention, axis=-1)
        out = attention @ v # B, N, N @ B, N, C
        y = mlp_attention[i]["mult_scalar"] * out + mlp_attention[i]["add_scalar"]
        
        # print(y.shape, "attention out")
        if activation is None:
            y = Activation('relu')(y)
        else:
            y = Activation(activation)(y)
    
    
    return y
    


def apply_mlp(x, mlp):
    y = x
    for i in range(len(mlp)):
        y = mlp[i]['layer'](y)
        y = mlp[i]['bn_layer'](y)
        y = Activation('relu')(y)
    return y

def set_sphere_weights(units, types):
    weights = dict()
    for l in types:
        if int(l) == 0:
            weights[l] = Dense(units=units)
        else:
            weights[l] = Dense(units=units, use_bias=False)
    return weights

def apply_layers(x, layers):
    y = dict()
    for l in x:
        if l.isnumeric():
            y[l] = layers[l](x[l])
    return y

def norms(x):
    y = []
    for l in x:
        if l.isnumeric():
            nxl = tf.reduce_sum(tf.multiply(x[l], x[l]), axis=-2, keepdims=False)
            y.append(nxl)
    n = tf.concat(y, axis=-1)
    n = tf.sqrt(tf.maximum(n, 1e-8))
    return n

def tf_fibonnacci_sphere_sampling(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    S2 = np.stack([x, y, z], axis=-1)
    return tf.convert_to_tensor(S2, dtype=tf.float32)

def stack_eq(x):
    Y = []
    for l in x:
        if l.isnumeric():
            Y.append(x[l])
    y = tf.concat(Y, axis=-2)
    y_shape = list(y.shape)
    y_shape = y_shape[:-1]
    y_shape[-1] = -1
    y = tf.reshape(y, y_shape)
    return y


def gauss_normalization(d, sigma):
    g = tf.exp(-(d*d) / (2.*sigma*sigma))
    g = tf.reduce_mean(g, axis=2, keepdims=True)
    g = tf.expand_dims(g, axis=-1)

    g = 1. / (g + 0.00000001)
    h = tf.reduce_mean(g, axis=1, keepdims=True)
    g = g / h
    return g

"""
class SphericalHarmonicsEval:
    def __init__(self, base='pentakis', l_max=3, l_list=None, sph_fn=None):
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

        if isinstance(base, str):
            S2 = tf_polyhedrons(self.base)
        else:
            S2 = base


        y = self.sph_fn.compute(S2)
        self.types = y.keys()
        Y = []
        for l in self.types:
            Y.append(tf.reshape(y[l], (-1, 2*int(l)+1)))
        self.Y = tf.concat(Y, axis=-1)

    def compute(self, x):
        X = []
        for l in self.types:
            X.append(x[l])
        X = tf.concat(X, axis=-2)
        return tf.einsum('vm,...mc->...vc', self.Y, X)
"""

def type_1(x, S2):
    y = SphericalHarmonicsCoeffs(l_list=[1], base=S2).compute(x)
    return y['1']

def zernike_monoms(x, max_deg):
    m = int(max_deg / 2.)
    n2 = tf.reduce_sum(x*x, axis=-1, keepdims=True)
    n2 = tf.expand_dims(n2, axis=-1)
    p = [tf.ones(n2.shape)]
    for m in range(m):
        p.append(p[-1]*n2)

    y = tf_spherical_harmonics(l_max=max_deg).compute(x)
    for l in y:
        y[l] = tf.expand_dims(y[l], axis=-1)

    z = dict()
    for d in range(max_deg+1):
        z[d] = []
    for l in y:
        l_ = int(l)
        for d in range(m+1):
            d_ = 2*d + l_
            if d_ <= max_deg:
                print(p[d].shape)
                print(y[l].shape)
                zd = tf.multiply(p[d], y[l])
                z[d_].append(zd)
    for d in z:
        z[d] = tf.concat(z[d], axis=-1)
    return z



class TFN(tf.keras.Model):
    def __init__(self, num_classes):
        super(TFN, self).__init__()
        self.num_classes = num_classes
        self.dodecahedron = 'pentakis'
        # self.dodecahedron = 'regular'
        self.d = 3
        self.l_max = [3, 3, 3]
        self.l_max_out = [3, 3, 3]
        self.num_shells = [3, 3, 3]
        self.gaussian_scale = []
        for i in range(len(self.num_shells)):
            self.gaussian_scale.append(0.69314718056 * ((self.num_shells[i]) ** 2))
        self.radius = [0.2, 0.40, 0.8]
        self.bounded = [True, True, True]
        # self.num_points = [1024, 512, 256, 64]
        # self.patch_size = [64, 64, 64]

        self.num_points = [1024, 256, 64, 16]
        self.patch_size = [32, 32, 32]

        self.spacing = [0, 0, 0]
        self.equivariant_units = [32, 64, 128]
        self.mlp_units = [[32, 32], [64, 64], [128, 256]]
        self.bn_momentum = 0.75
        self.droupout_rate = 0.5

        self.grouping_layers = []
        self.kernel_layers = []
        self.conv_layers = []
        self.eval = []
        self.coeffs = []

        for i in range(len(self.radius)):
            gi = GroupPoints(radius=self.radius[i],
                             patch_size_source=self.patch_size[i],
                             spacing_source=self.spacing[i])
            self.grouping_layers.append(gi)

            ki = SphericalHarmonicsGaussianKernels(l_max=self.l_max[i],
                                                   gaussian_scale=self.gaussian_scale[i],
                                                   num_shells=self.num_shells[i],
                                                   bound=self.bounded[i])
            ci = ShGaussianKernelConv(l_max=self.l_max[i], l_max_out=self.l_max_out[i])

            self.kernel_layers.append(ki)
            self.conv_layers.append(ci)

            # self.eval.append(DodecahedronEval(l_max=self.l_max_out[i], dodecahedron=self.dodecahedron))
            # self.coeffs.append(DodecahedronCoeffs(l_max=self.l_max_out[i], dodecahedron=self.dodecahedron))

        self.mlp = []
        self.equivariant_weights = []
        self.bn = []

        for i in range(len(self.radius)):
            self.bn.append(BatchNormalization(momentum=self.bn_momentum))
            types = [str(l) for l in range(self.l_max_out[i] + 1)]
            self.equivariant_weights.append(set_sphere_weights(self.equivariant_units[i], types=types))
            self.mlp.append(set_mlp(self.mlp_units[i], self.bn_momentum))

        self.mlp_sphere = set_mlp(units=[128, 256, 512], momentum=self.bn_momentum)
        self.mlp_shape = set_mlp(units=[128, 256, 512], momentum=self.bn_momentum)

        self.fc1_units = 512
        self.fc2_units = 256

        self.fc1 = Dense(units=self.fc1_units, activation=None)
        self.bn_fc1 = BatchNormalization(momentum=self.bn_momentum)
        self.fc2 = Dense(units=self.fc2_units, activation=None)
        self.bn_fc2 = BatchNormalization(momentum=self.bn_momentum)
        self.softmax = Dense(units=self.num_classes, activation='softmax')
        self.S2 = tf_fibonnacci_sphere_sampling(64)

        self.basis_dim = 3
        self.basis_mlp = set_mlp([64], momentum=self.bn_momentum)
        self.basis_layer = Dense(units=self.basis_dim)

        self.code_dim = 64
        self.code_mlp = set_mlp([128], momentum=self.bn_momentum)
        self.code_layer = Dense(units=self.code_dim)

        # self.points_inv_mlp = set_mlp(units=[32], momentum=self.bn_momentum)
        self.points_inv_mlp = set_mlp_attention(units=[32], momentum=self.bn_momentum)
        # print(self.points_inv_mlp.layers)
        self.points_inv_layer = Dense(units=self.basis_dim)

        self.capsules_mlp = set_mlp(units=[256, 128, 16], momentum=self.bn_momentum)

    def call(self, x):

        if x.shape[1] is None:
            # This avoids the error for multi dimensions :))))
            B = x.shape[0]
            x = tf.random.normal((B, 1024, 3))
        points = [x]
        grouped_points = []
        kernels = []

        num_points_ = self.num_points
        
        num_points_[0] = x.shape[1]

        for i in range(len(self.radius)):
            pi = kd_pooling_1d(points[-1], int(num_points_[i] / num_points_[i + 1]))
            # pi = Jitter(self.jitter_scale[i])(pi)
            points.append(pi)

        yzx = []
        for i in range(len(points)):
            yzx_i = tf.stack([points[i][..., 1], points[i][..., 2], points[i][..., 0]], axis=-1)
            yzx.append(tf.expand_dims(yzx_i, axis=-1))

        for i in range(len(self.radius)):
            gi = self.grouping_layers[i]({"source points": points[i], "target points": points[i + 1]})
            ki = self.kernel_layers[i]({"patches": gi["patches source"], "patches dist": gi["patches dist source"]})
            grouped_points.append(gi)
            kernels.append(ki)

        y = {'0': tf.ones((x.shape[0], x.shape[1], 1, 1))}
        for i in range(len(self.radius)):
            y["source points"] = points[i]
            y["target points"] = points[i + 1]
            y["patches idx"] = grouped_points[i]["patches idx source"]
            y["patches dist source"] = grouped_points[i]["patches dist source"]
            y["kernels"] = kernels[i]

            if '1' in y:
                y['1'] = tf.concat([y['1'], yzx[i]], axis=-1)
            else:
                y['1'] = yzx[i]

            y = self.conv_layers[i](y)

            if '1' in y:
                y['1'] = tf.concat([y['1'], yzx[i + 1]], axis=-1)
            else:
                y['1'] = yzx[i + 1]

            y = apply_layers(y, self.equivariant_weights[i])
            y = SphericalHarmonicsEval(l_max=self.l_max_out[i], base=self.S2).compute(y)
            y = self.bn[i](y)
            y = Activation('relu')(y)
            y = apply_mlp(y, self.mlp[i])
            if i < len(self.radius) - 1:
                y = SphericalHarmonicsCoeffs(l_max=self.l_max_out[i], base=self.S2).compute(y)



        y = tf.reduce_max(y, axis=1, keepdims=False)

        y_ = y
        basis = apply_mlp(y_, self.basis_mlp)
        basis = self.basis_layer(basis)
        basis = type_1(basis, self.S2)
        basis = tf.linalg.l2_normalize(basis, epsilon=1e-4, axis=-2)

        latent_code = apply_mlp(y, self.code_mlp)
        latent_code = self.code_layer(latent_code)
        latent_code = SphericalHarmonicsCoeffs(l_max=self.l_max_out[-1], base=self.S2).compute(latent_code)

        z = zernike_monoms(points[0], self.l_max_out[-1])
        points_code = []
        for l in latent_code:
            # Invariant feature space (H matrix) H = F.T @ z
            p = tf.einsum('bmi,bvmj->bvij', latent_code[l], z[int(l)])
            shape = list(p.shape)
            shape = shape[:-1]
            shape[-1] = -1
            p = tf.reshape(p, shape)
            points_code.append(p)

        points_code = tf.concat(points_code, axis=-1)

        capsules = 2.*apply_mlp(points_code, self.capsules_mlp)
        capsules = tf.nn.softmax(capsules, axis=-1)
        # capsules_mean = tf.reduce_mean(capsules, axis=1, keepdims=True)
        # eps = 1e-6
        # capsules = tf.divide(capsules, capsules_mean + eps)

        points_inv = apply_mlp_attention(points_code, self.points_inv_mlp)
        points_inv = self.points_inv_layer(points_inv)

        return capsules, points_inv, basis











