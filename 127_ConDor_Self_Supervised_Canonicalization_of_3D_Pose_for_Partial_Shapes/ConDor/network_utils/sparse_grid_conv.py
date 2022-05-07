import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Concatenate, Add
from spherical_harmonics.kernels import ZernikeKernelConv
from group_points import GroupPoints
from pooling import GridBatchSampler, GridBatchPooling
from activations import EquivariantActivationModule, L2Normalize
from spherical_harmonics.kernels import zernike_kernel_3D, tf_monomial_basis_3D_idx, tf_monomial_basis_coeffs, \
    tf_gaussian_polynomial_1d_grid_kernel, tf_zernike_kernel_basis
from spherical_harmonics.kernels import tf_clebsch_gordan_decomposition
from sympy import *

# CLASSICAL (NON FRACTIONAL) POOLING AND STRIDED CONV CAN BE DONE WITH TF WHERE + SORT + CROP
# WHERE TO SELECT INDICES AND REPLACE UNWANTED ONES WITH -1
# SORT TO HAVE ALL -1 AT END OR BEGINING
# CROP TO REMOVE -1 s
def grid_conv_1d_idx(table, grid_size, linear_idx, size, dilation):
    if dilation is None:
        dilation = [1]*len(size)
    idx = []
    offset = grid_size**(len(size)-1)
    for i in range(len(size)):
        idx_i = tf.expand_dims((offset*dilation[i])*tf.range(-size[i], size[i]), axis=0)
        idx_i = tf.add(tf.expand_dims(linear_idx, axis=-1), idx_i)
        idx.append(table.lookup(idx_i))
        offset = int(offset / grid_size)
    return idx


def separable_grid_conv(x, idx, kernel):
    x_shape = x.shape
    y = tf.reshape(x, (x.shape[0], 1, -1, 1))
    y = tf.tile(y, (1, kernel))
    for i in range(len(kernel)):
        y = tf.gather(params=y, indices=idx[i], axis=0)
        y = tf.nn.conv2d(y, kernel[i], padding='VALID')
        y = y[:, 0, ...]
    y = tf.transpose(y, (0, 2, 1))
    y = tf.reshape(y, (x_shape[0], -1, x_shape[-2], x_shape[-1]))
    return y



class SeparableConv(tf.keras.layers.Layer):
    def __init__(self, kernels, dilation=None):
        super(SeparableConv, self).__init__()
        self.kernels = kernels
        if dilation is None:
            self.dilation = [1] * len(kernels)
        else:
            self.dilation = dilation
        self.size = []
        for i in range(len(kernels)):
            self.size.append(int(kernels[0].shape[0] / 2))

    def build(self, input_shape):
        super(SeparableConv, self).build(input_shape)

    def call(self, x):
        assert (isinstance(x, dict))
        linear_idx = x["linear idx"]
        table = x["lookup table"]
        grid_size = x["grid size"]
        idx = grid_conv_1d_idx(table, grid_size, linear_idx, self.size, self.dilation)
        return separable_grid_conv(x["signal"], idx, self.kernels)


class ZernikeGaussianGridConv(tf.keras.layers.Layer):
    def __init__(self, gaussian_scale, d, kernel_size, l_max_out=None, dilation=None):
        super(ZernikeGaussianGridConv, self).__init__()
        self.l_max_out = l_max_out
        self.gaussian_scale = gaussian_scale
        self.d = d
        self.kernel_size = kernel_size
        if dilation is None:
            self.dilation = [1]*3
        else:
            if isinstance(dilation, list):
                self.dilation = dilation
            else:
                self.dilation = [dilation] * 3

        k = tf_gaussian_polynomial_1d_grid_kernel(gaussian_scale, d, kernel_size,
                                                  per_bin=10, kernel_scale=1., dtype=tf.float32)
        monoms_idx = tf_monomial_basis_3D_idx(d)
        self.monomial_basis_size = monoms_idx.shape[0]
        # monoms_idx = tf.transpose(monoms_idx, (1, 0))
        k = tf.reshape(k, (1, k.shape[0], k.shape[1], 1))
        k = tf.gather(k, monoms_idx, axis=2)
        self.monomial_kernels = [k[..., 0, :], k[..., 1, :], k[..., 2, :]]

        # compute zernike basis and normalization of kernels
        kx = tf.reshape(self.monomial_kernels[0], (k.shape[1], 1, 1, k.shape[2]))
        ky = tf.reshape(self.monomial_kernels[1], (1, k.shape[1], 1, k.shape[2]))
        kz = tf.reshape(self.monomial_kernels[2], (1, 1, k.shape[1], k.shape[2]))
        kernels = tf.multiply(tf.multiply(kx, ky), kz)

        Z = tf_zernike_kernel_basis(d)
        self.zernike_split_size = []
        self.Z = []
        for l in range(len(Z)):
            k_abs_l = tf.einsum('nmj,xyzj->xyznm', Z[l], kernels)
            k_abs_l = tf.reduce_sum(tf.abs(k_abs_l), axis=[0, 1, 2], keepdims=False)
            k_abs_l = tf.reduce_mean(k_abs_l, axis=-1, keepdims=False)
            k_abs_l = tf.maximum(k_abs_l, 0.00000001)
            k_abs_l = tf.reshape(k_abs_l, (-1, 1, 1))
            Z[l] = tf.divide(Z[l], k_abs_l)
            self.zernike_split_size.append(Z[l].shape[0]*Z[l].shape[1])
            self.Z.append(tf.reshape(Z[l], (-1, Z[l].shape[-1])))
        self.Z = tf.concat(self.Z, axis=0)

    def build(self, input_shape):
        assert(isinstance(input_shape, dict))
        self.l_max_in = 0
        for l in input_shape:
            if l.isnumeric():
                self.l_max_in = max(self.l_max_in, int(l))
        self.l_max_in = max(self.l_max_in, self.d)
        # Clebsch Gordan
        self.Q = tf_clebsch_gordan_decomposition(l_max=self.l_max_in, sparse=False, l_max_out=self.l_max_out)
        super(ZernikeGaussianGridConv, self).build(input_shape)

    def call(self, x):
        assert (isinstance(x, dict))
        linear_idx = x["linear idx"]
        table = x["lookup table"]
        grid_size = x["grid size"]
        idx = grid_conv_1d_idx(table, grid_size, linear_idx, self.size, self.dilation)
        signal = []
        features_type = []
        channels_split_size = []
        for l in x:
            if l.isnumeric():
                features_type.append(int(l))
                # channels_split_size .append(x[l].shape[-2]*x[l].shape[-1])
                # signal.append(tf.reshape(x[l], (x[l].shape[0], -1)))
                channels_split_size.append(self._build_input_shape[l][-2]*self._build_input_shape[l][-1])
                signal.append(tf.reshape(x[l], (self._build_input_shape[l][0], -1)))

        signal = tf.concat(signal, axis=-1)
        signal = tf.expand_dims(signal, axis=1)
        y = tf.tile(signal, (1, self.monomial_basis_size, 1))
        for i in range(3):
            y = tf.gather(params=y, indices=idx[i], axis=0)
            # try tanspose + matmul as well
            y = tf.reduce_sum(tf.multiply(self.monomial_kernels[i], y), axis=1, keepdims=False)
        y = tf.matmul(self.Z, y)
        # split y
        y_ = tf.split(y, num_or_size_splits=channels_split_size, axis=-1)
        y = {str(j):[] for j in range(self.d+1)}
        y_cg = []
        for i in range(len(channels_split_size)):
            l = features_type[i]
            # yi = tf.reshape(y[i], (self._build_input_shape[str(l)][0], -1, self._build_input_shape[str(l)][-1]))
            yi = tf.reshape(y_[i], (self._build_input_shape[str(l)][0], -1, 2*l+1, self._build_input_shape[str(l)][-1]))
            yi = tf.transpose(yi, (0, 2, 1, 3))
            yi = tf.split(yi, num_or_size_splits=self.zernike_split_size, axis=2)
            for j in range(len(self.zernike_split_size)):
                # yij = tf.transpose(yi[j], (0, 2, 1, 3))
                yij = tf.reshape(yi[j], (self._build_input_shape[str(l)][0], 2*l+1, 2*j+1, -1))
                if l == 0:
                    y[j].append(yij[:, 0, :, :])
                elif j == 0:
                    y[l].append(yij[:, :, 0, :])
                else:
                    y_cg.append(yij)

        y_cg = self.Q.decompose(y_cg)
        for J in y_cg:
            if J not in y:
                y[J] = []
            y[J].append(y_cg[J])
        for J in y:
            y[J] = tf.concat(y[J], axis=-1)
        return y





