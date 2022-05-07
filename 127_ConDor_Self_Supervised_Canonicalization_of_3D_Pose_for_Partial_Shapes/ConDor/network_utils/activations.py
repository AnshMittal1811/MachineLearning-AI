import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Concatenate, Reshape
import numpy as np
from spherical_harmonics.kernels import tf_spherical_harmonics, tf_eval_monoms
from spherical_harmonics.tf_spherical_harmonics import normalized_real_sh
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import backend as K

def np_regular_dodecahedron():
    C0 = (1 + np.sqrt(5)) / 4
    C1 = (3 + np.sqrt(5)) / 4

    V = np.array([[0.0, 0.5, C1], [0.0, 0.5, -C1], [0.0, -0.5, C1], [0.0, -0.5, -C1],
                  [C1, 0.0, 0.5], [C1, 0.0, -0.5], [-C1, 0.0, 0.5], [-C1, 0.0, -0.5],
                  [0.5, C1, 0.0], [0.5, -C1, 0.0], [-0.5, C1, 0.0], [-0.5, -C1, 0.0],
                  [C0, C0, C0], [C0, C0, -C0], [C0, -C0, C0], [C0, -C0, -C0],
                  [-C0, C0, C0], [-C0, C0, -C0], [-C0, -C0, C0], [-C0, -C0, -C0]], dtype=np.float32)

    F = np.array([[0, 2, 14, 4, 12], [0, 12, 8, 10, 16], [0, 16, 6, 18, 2], [7, 6, 16, 10, 17],
                  [7, 17, 1, 3, 19], [7, 19, 11, 18, 6], [9, 11, 19, 3, 15], [9, 15, 5, 4, 14],
                  [9, 14, 2, 18, 11], [13, 1, 17, 10, 8], [13, 8, 12, 4, 5], [13, 5, 15, 3, 1]], dtype=np.int32)
    return V, F


def np_pentakis_dodecahedron():
    C0 = 3 * (np.sqrt(5) - 1) / 4
    C1 = 9 * (9 + np.sqrt(5)) / 76
    C2 = 9 * (7 + 5 * np.sqrt(5)) / 76
    C3 = 3 * (1 + np.sqrt(5)) / 4

    V = np.array([[0.0, C0, C3], [0.0, C0, -C3], [0.0, -C0, C3], [0.0, -C0, -C3],
                  [C3, 0.0, C0], [C3, 0.0, -C0], [-C3, 0.0, C0], [-C3, 0.0, -C0],
                  [C0, C3, 0.0], [C0, -C3, 0.0], [-C0, C3, 0.0], [-C0, -C3, 0.0],
                  [C1, 0.0, C2], [C1, 0.0, -C2], [-C1, 0.0, C2], [-C1, 0.0, -C2],
                  [C2, C1, 0.0], [C2, -C1, 0.0], [-C2, C1, 0.0], [-C2, -C1, 0.0],
                  [0.0, C2, C1], [0.0, C2, -C1], [0.0, -C2, C1], [0.0, -C2, -C1],
                  [1.5, 1.5, 1.5], [1.5, 1.5, -1.5], [1.5, -1.5, 1.5], [1.5, -1.5, -1.5],
                  [-1.5, 1.5, 1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5], [-1.5, -1.5, -1.5]],
                 dtype=np.float)

    F = np.array([[12,  0,  2], [12,  2, 26], [12, 26,  4], [12,  4, 24],
                  [12, 24,  0], [13,  3,  1], [13,  1, 25], [13, 25,  5],
                  [13,  5, 27], [13, 27,  3], [14,  2,  0], [14,  0, 28],
                  [14, 28,  6], [14,  6, 30], [14, 30,  2], [15,  1,  3],
                  [15,  3, 31], [15, 31,  7], [15,  7, 29], [15, 29,  1],
                  [16,  4,  5], [16,  5, 25], [16, 25,  8], [16,  8, 24],
                  [16, 24,  4], [17,  5,  4], [17,  4, 26], [17, 26,  9],
                  [17,  9, 27], [17, 27,  5], [18,  7,  6], [18,  6, 28],
                  [18, 28, 10], [18, 10, 29], [18, 29,  7], [19,  6,  7],
                  [19,  7, 31], [19, 31, 11], [19, 11, 30], [19, 30,  6],
                  [20,  8, 10], [20, 10, 28], [20, 28,  0], [20,  0, 24],
                  [20, 24,  8], [21, 10,  8], [21,  8, 25], [21, 25,  1],
                  [21,  1, 29], [21, 29, 10], [22, 11,  9], [22,  9, 26],
                  [22, 26,  2], [22,  2, 30], [22, 30, 11], [23,  9, 11],
                  [23, 11, 31], [23, 31,  3], [23,  3, 27], [23, 27,  9]],
                 dtype=np.int32)
    return V, F

def tf_pentakis_dodecahedron():
    V, F = np_pentakis_dodecahedron()
    V = tf.convert_to_tensor(V, dtype=tf.float32)
    F = tf.convert_to_tensor(F, dtype=tf.int32)
    return V, F

def tf_regular_dodecahedron():
    V, F = np_regular_dodecahedron()
    V = tf.convert_to_tensor(V, dtype=tf.float32)
    F = tf.convert_to_tensor(F, dtype=tf.int32)
    return V, F

def tf_dodecahedron_sph(l_max, dodecahedron='regular'):
    if dodecahedron == 'pentakis':
        V, _ = tf_pentakis_dodecahedron()
    else:
        V, _ = tf_regular_dodecahedron()
    V = tf.linalg.l2_normalize(V, axis=-1)
    L = range(l_max + 1)
    coeffs, idx, monoms_basis_idx = tf_spherical_harmonics(L)
    """
    P = dict()
    for l in L:
        monoms = tf_eval_monoms(V, monoms_idx=monoms_basis_idx)
        P[str(l)] = tf.einsum('mi,vmi->vm', coeffs[l], tf.gather(monoms, idx[l], axis=-1))
    """
    P = []
    for l in L:
        monoms = tf_eval_monoms(V, monoms_idx=monoms_basis_idx)
        P.append(tf.einsum('mi,vmi->vm', coeffs[l], tf.gather(monoms, idx[l], axis=-1)))
    return tf.concat(P, axis=-1)

class DodecahedronCoeffs(tf.keras.layers.Layer):
    def __init__(self, l_max, dodecahedron='regular'):
        self.l_max = l_max
        self.dodecahedron = dodecahedron
        self.P = tf_dodecahedron_sph(l_max, dodecahedron)
        if dodecahedron == 'regular':
            self.num_vertices = 20
        else:
            self.num_vertices = 32

        self.split_size = []
        for l in range(l_max+1):
            self.split_size.append(2*l+1)

        super(DodecahedronCoeffs, self).__init__()

    def build(self, input_shape):
        super(DodecahedronCoeffs, self).build(input_shape)

    def call(self, x):
        """
        coeffs = dict()
        for l in self.P:
            # coeffs[l] = tf.einsum('im,bvic->bvmc', self.P[l], x)
            Pl = tf.reshape(self.P[l], [1] * (len(list(x.shape)) - 2) + list(self.P[l].shape))
            coeffs[l] = tf.matmul(Pl, x, transpose_a=True)
        """
        coeffs = dict()
        # P = tf.reshape(self.P, [1] * (len(list(x.shape)) - 2) + list(self.P.shape))
        # c = tf.matmul(P, x, transpose_a=True)
        # c = tf.zeros(list(x.shape)[:-2] + [P.shape[-1]] + [x.shape[-1]])
        print(x.shape)
        # c = tf.einsum('im,bvic->bvmc', self.P, x)
        print(self.P)
        print(x)
        c = tf.matmul(tf.transpose(self.P), x)
        c = c / float(self.num_vertices / (4.*np.pi))
        c = tf.split(c, num_or_size_splits=self.split_size, axis=-2)

        p = 0
        for l in range(self.l_max+1):
            # coeffs[str(l)] = c[..., p:p+2*l+1, :]
            coeffs[str(l)] = c[l]
            p += 2*l+1
        return coeffs



class DodecahedronEval(tf.keras.layers.Layer):
    def __init__(self, l_max, dodecahedron='regular'):
        self.l_max = l_max
        self.dodecahedron = dodecahedron
        self.P = tf_dodecahedron_sph(l_max, dodecahedron)
        super(DodecahedronEval, self).__init__()

    def build(self, input_shape):
        super(DodecahedronEval, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, dict)
        """
        y = []
        for l in x:
            if l.isnumeric() and l in self.P:
                # yl = tf.einsum('im,bvmc->bvic', self.P[l], x)
                Pl = tf.reshape(self.P[l], [1] * (len(list(x[l].shape)) - 2) + list(self.P[l].shape))
                y.append(tf.matmul(Pl, x[l]))
        return tf.add_n(y)
        """
        y = [None]*(self.l_max+1)
        for l in x:
            if l.isnumeric():
                if int(l) <= self.l_max:
                    y[int(l)] = x[l]
        y = tf.concat(y, axis=-2)
        # P = tf.reshape(self.P, [1] * (len(list(y.shape)) - 2) + list(self.P.shape))
        # return tf.ones(list(y.shape[:-2]) + [self.P.shape[-2]] + [y.shape[-1]])
        # return tf.einsum('im,bvmc->bvic', self.P, y)
        return tf.matmul(self.P, y)




def norml2(x, axis=-1, keepdims=False, eps=0.0000001):
    y = tf.multiply(x, x)
    y = tf.reduce_sum(y, axis=axis, keepdims=keepdims)
    y = tf.sqrt(tf.maximum(y, eps))
    return y

class L2Norms(tf.keras.layers.Layer):
    def __init__(self, axis=-2, eps=0.000001, keepdims=True):
        self.axis = axis
        self.eps = eps
        self.keepdims = keepdims
        super(L2Norms, self).__init__()

    def build(self, input_shape):
        super(L2Norms, self).build(input_shape)

    def call(self, x):
        if isinstance(x, dict):
            y = dict()
            for l in x:
                # if isinstance(l, int):
                if l.isnumeric():
                    y[l] = norml2(x[l], axis=self.axis, eps=self.eps, keepdims=self.keepdims)
        else:
            return norml2(x, axis=self.axis, eps=self.eps, keepdims=self.keepdims)

def squash(x, b=1., eps=0.0000001, bias=None):
    n2 = tf.reduce_sum(tf.multiply(x, x), axis=-2, keepdims=True)
    n = tf.sqrt(tf.maximum(n2, eps))
    if bias is not None:
        n = tf.nn.relu(tf.subtract(n, bias))
    a = tf.divide(n, n2 + b)
    return tf.multiply(a, x)

class Squash(tf.keras.layers.Layer):
    def __init__(self, b=1., eps=0.0000001, use_bias=False):
        self.b = b
        self.eps = eps
        self.use_bias = use_bias
        super(Squash, self).__init__()

    def build(self, input_shape):
        self.bias = None
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[1, 1, 1, input_shape[-1]],
                                        initializer='zero',
                                        regularizer=None,
                                        constraint=None,
                                        dtype=tf.float32,
                                        trainable=True)
        super(Squash, self).build(input_shape)

    def call(self, x):
        if isinstance(x, dict):
            y = dict()
            for l in x:
                if l.isnumeric():
                    y[l] = squash(x[l], b=self.b, eps=self.eps, bias=self.bias)
        else:
            return squash(x, b=self.b, eps=self.eps, bias=self.bias)

def norm_activation(x, b=0.00000001, eps=0.0000001, bias=None):
    l = x.shape[-2]
    if l > 0:
        n2 = tf.reduce_sum(tf.multiply(x, x), axis=-2, keepdims=True)
        n = tf.sqrt(tf.maximum(n2, eps))
        if bias is not None:
            n = tf.nn.relu(tf.subtract(n, bias))
        a = tf.divide(n, n2 + b)
        return tf.multiply(a, x)
    else:
        return tf.nn.relu(x)

class NormActivation(tf.keras.layers.Layer):
    def __init__(self, b=0.00000001, eps=0.0000001, use_bias=False):
        self.b = b
        self.eps = eps
        self.use_bias = use_bias
        super(NormActivation, self).__init__()

    def build(self, input_shape):
        self.bias = None
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[1, 1, 1, input_shape[-1]],
                                        initializer='zero',
                                        regularizer=None,
                                        constraint=None,
                                        dtype=tf.float32,
                                        trainable=True)
        super(NormActivation, self).build(input_shape)

    def call(self, x):
        if isinstance(x, dict):
            y = dict()
            for l in x:
                if l.isnumeric():
                    y[l] = squash(x[l], b=self.b, eps=self.eps, bias=self.bias)
        else:
            return squash(x, b=self.b, eps=self.eps, bias=self.bias)

class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, relu=True, *args, **kwargs):
        self.relu = relu
        self.units = units
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        shape = [1]*len(input_shape)
        shape[-1] = input_shape[-1]
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        self.bias = self.add_weight('bias',
                                    shape=[self.units],
                                    initializer='zeros',
                                    dtype=K.floatx(),
                                    regularizer=None,
                                    constraint=None,
                                    trainable=False)
        self.built = True
    def call(self, x):
        # y = tf.add(self.bias, x)
        y = tf.nn.bias_add(x, self.bias)
        if self.relu:
            y = tf.nn.relu(y)
        return y

def set_norm_activation_layers_sphere(units, l_max, momentum):
    w = dict()
    bn = dict()
    b = dict()
    for l in range(l_max+1):
        w[str(l)] = Dense(units=units, use_bias=False)
        bn[str(l)] = BatchNormalization(momentum=momentum)
        b[str(l)] = BiasLayer(units=units)
        # b[str(l)] = Dense(units=units, use_bias=True)
    return {'weights': w, 'batch_norm': bn, 'bias': b}




def apply_norm_layer(x, w):
    y = dict()
    for l in x:
        if l.isnumeric():
            if int(l) == 0:
                yl = w['weights'][l](x[l])
                yl = w['batch_norm'][l](yl)
                yl = w['bias'][l](yl)
                y[l] = yl
            else:
                yl = w['weights'][l](x[l])
                n2 = tf.multiply(yl, yl)
                n2 = tf.reduce_sum(n2, axis=-2, keepdims=True)
                n = tf.sqrt(tf.maximum(n2, 0.00001))
                nn = w['batch_norm'][l](n)
                nn = w['bias'][l](nn)
                a = tf.divide(nn, n)
                yl = tf.multiply(a, yl)
                # yl = tf.divide(yl, n)
                # yl = tf.multiply(nn, yl)
                y[l] = yl
    return y

def set_gater_layer(units, l_max, momentum):
    w = dict()
    w['bn'] = dict()
    w['weights'] = dict()
    w['gate_weights'] = dict()
    w['gate_bn'] = dict()

    for l in range(l_max+1):
        key = str(l)
        w['bn'][key] = BatchNormalization(momentum=momentum)
        use_bias = (l == 0)
        w['weights'][key] = Dense(units=units, use_bias=use_bias)


    for l in range(1,l_max+1):
        key = str(l)
        w['gate_bn'][key] = BatchNormalization(momentum=momentum)
        w['gate_weights'][key] = Dense(units=units)

    return w

def apply_gated_layer(x, w):
    # apply weights to invariants to learn non linearities

    y = dict()

    y0 = w['weights']['0'](x['0'])
    y0 = w['bn']['0'](y0)
    y0 = tf.nn.relu(y0)
    y['0'] = y0

    for l in x:
        if l.isnumeric():
            if int(l) > 0:
                gl = w['gate_weights'][l](x['0'])
                gl = w['gate_bn'][l](gl)
                gl = tf.sigmoid(gl)
                yl = w['weights'][l](x[l])
                yl = tf.multiply(gl, yl)
                y[l] = yl
    return y





def tf_l2_normalize(x, axis=-1, eps=0.000001, denominator='max'):
    if denominator == 'avg':
        n = tf.sqrt(tf.maximum(tf.reduce_sum(tf.multiply(x, x), axis=axis, keepdims=True), 0.000001))
        return tf.divide(x, n + eps)
    else:
        return tf.math.l2_normalize(x, axis=axis, epsilon=eps)


class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, axis=-1, eps=0.000001, denominator='max'):
        self.denominator = denominator
        self.eps = eps
        self.axis = axis
        super(L2Normalize, self).__init__()

    def build(self, input_shape):
        super(L2Normalize, self).build(input_shape)

    def call(self, x):
        if isinstance(x, dict):
            for l in x:
                x[l] = tf_l2_normalize(x[l], axis=self.axis, eps=self.eps, denominator=self.denominator)
            return x
        else:
            return tf_l2_normalize(x, axis=self.axis, eps=self.eps, denominator=self.denominator)


"""
class NormActivation(tf.keras.layers.Layer):
    def __init__(self, normalize=False, eps=1):
        self.normalize = normalize
        self.eps = eps
        super(DotProd, self).__init__()

    def build(self, input_shape):
        self
        pass

    def call(self, x):
        assert (isinstance(x, list))
        assert (len(x) % 2 == 0)
        k = int(len(x) / 2)
        dot_prods = []
        for i in range(k):
            ti = x[i]
            ui = x[i+k]
            if self.normalize:
                ui = tf.math.l2_normalize(ui, axis=-2, epsilon=self.eps)
            di = tf.einsum('...ij,...ik->...jk', ti, ui)
            s = list(di.shape)
            s.pop()
            s[-1] = -1
            di = tf.reshape(di, s)
            dot_prods.append(di)
        return tf.concat(dot_prods, axis=-1)
"""

class DotProd(tf.keras.layers.Layer):
    def __init__(self, normalize=False, eps=0.01):
        self.normalize = normalize
        self.eps = eps
        super(DotProd, self).__init__()

    def build(self, input_shape):
        super(DotProd, self).build(input_shape)

    def call(self, x):
        assert (isinstance(x, list))
        assert (len(x) % 2 == 0)
        k = int(len(x) / 2)
        dot_prods = []
        for i in range(k):
            ti = x[i]
            ui = x[i+k]
            if self.normalize:
                ui = tf.math.l2_normalize(ui, axis=-2, epsilon=self.eps)
            di = tf.einsum('...ij,...ik->...jk', ti, ui)
            s = list(di.shape)
            s.pop()
            s[-1] = -1
            di = tf.reshape(di, s)
            dot_prods.append(di)
        return tf.concat(dot_prods, axis=-1)

class EquivariantActivationModule(tf.keras.Model):

    def __init__(self, units_invar, units_eqvar=None, size_eqvar_basis=None, activation='relu',
                 bn_momentum=0.99, bn_epsilon=0.001, use_bn=True, normalize=True, eps=0.01):
        super(EquivariantActivationModule, self).__init__()
        self.invar_units = units_invar
        self.eqvar_units = units_eqvar
        self.size_eqvar_basis = size_eqvar_basis
        self.activation = activation
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.use_bn = use_bn
        self.normalize = normalize
        self.eps = eps

        if size_eqvar_basis is None:
            self.eqvar_units = None
        else:
            assert(isinstance(size_eqvar_basis, list))
            if len(size_eqvar_basis) == 0:
                self.size_eqvar_basis = None
                self.eqvar_units = None

        assert((self.invar_units is not None) or (self.eqvar_units is not None))
        self.num_eqvar_heads_out = len(size_eqvar_basis)
        if self.invar_units is not None:
            if self.use_bn:
                self.bn_invar = BatchNormalization(momentum=self.bn_momentum, epsilon=self.bn_epsilon)
            else:
                self.bn_invar = None
            self.dense_invar = Dense(units=self.invar_units)
        else:
            self.bn_invar = None
            self.dense_invar = None

        if self.eqvar_units is not None:
            self.dense_eqvar = []
            self.bn_eqvar = []
            for i in range(self.num_eqvar_heads_out):
                self.dense_eqvar.append(Dense(units=self.eqvar_units[i]*size_eqvar_basis))
                self.bn_eqvar.append(BatchNormalization(momentum=self.bn_momentum, epsilon=self.bn_epsilon))
        else:
            self.dense_eqvar = None
            self.bn_eqvar = None

    def call(self, inputs):
        invar = None
        eqvar = None

        output = dict()

        if self.eqvar_units is not None:
            assert(isinstance(inputs, list))
            if (len(inputs) - len(self.size_eqvar_basis)) % 2 == 0:
                num_invar = 0
            else:
                num_invar = 1
            num_eqvar_heads_in = int((len(inputs) - len(self.size_eqvar_basis)) / 2)
            V = inputs[self.num_invar + 2*self.num_eqvar_heads_in:]
            dot_prods = DotProd(normalize=self.normalize,
                                eps=self.eps)(inputs[num_invar:num_invar+2*num_eqvar_heads_in])
            invar_features = dot_prods
            if num_invar == 1:
                invar_features = Concatenate()(inputs[0][..., 0, :], dot_prods)
            if self.invar_units is not None:
                invar = self.dense_invar(invar_features)

            for l in range(self.num_eqvar_heads_out):
                wl = self.dense_eqvar[l](Concatenate()(invar_features))
                if self.bn_eqvar is not None:
                    wl = self.bn_eqvar[l](wl)
                if self.bn_invar is not None:
                    wl = Activation(self.activation)(wl)
                wl_shape = list(wl.shape)
                wl_shape.append(int(wl_shape[-1] / self.eqvar_units[l]))
                wl_shape[-2] = self.eqvar_units[l]
                wl = tf.reshape(wl, wl_shape)
                vl = tf.einsum('...ij,...mj->mi', wl, V[l])
                J = V[l].shape[-2]
                if J not in output:
                    output[J] = []
                output[J].append(vl)
        else:
            if self.invar_units is not None:
                invar = self.dense_invar(inputs[0])

        if invar is not None:
            if self.bn_invar is not None:
                invar = self.bn_invar(invar)
            if self.activation is not None:
                invar = Activation(self.activation)(invar)

        if eqvar is not None:
            if 0 not in output:
                output[0] = []
            output[0].append(invar)
            for l in output:
                output[l] = Concatenate()(output[l])
            return output
        else:
            return {0: invar}

"""
class EquivariantActivationModule(tf.keras.Model):

    def __init__(self, units_invar, units_eqvar=None, activation='relu',
                 bn_momentum=0.99, bn_epsilon=0.001, use_bn=True):
        super(EquivariantActivationModule, self).__init__()
        self.invar_units = units_invar
        self.eqvar_units = units_eqvar
        self.activation = activation
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.use_bn = use_bn

    def build(self, input_shape):
        self.num_eqvar_heads_in = 0
        self.num_eqvar_heads_out = 0
        self.num_invar = 0
        assert ( (self.invar_units is not None) or (self.eqvar_units is not None) )
        if self.invar_units is not None:
            if self.use_bn:
                self.bn_invar = BatchNormalization(momentum=self.bn_momentum, epsilon=self.bn_epsilon)
            else:
                self.bn_invar = None
            self.dense_invar = Dense(units=self.invar_units)
        else:
            self.bn_invar = None
            self.dense_invar = None

        if self.eqvar_units is not None:
            assert (isinstance(input_shape, list))
            self.dense_eqvar = []
            self.bn_eqvar = []
            if self.invar_units is not None:
                num_eqvar_heads_in = len(input_shape) - len(self.eqvar_units) - 1
                self.num_invar = 1
            else:
                num_eqvar_heads_in = len(input_shape) - len(self.eqvar_units)
                self.num_invar = 0
            assert(num_eqvar_heads_in % 2 == 0)
            self.num_eqvar_heads_in = int(num_eqvar_heads_in / 2)
            self.num_eqvar_heads_out = len(input_shape) - self.num_invar - self.num_eqvar_heads_in
            assert(len(self.eqvar_units) == self.num_eqvar_heads_out)
            offset = self.num_invar + 2*self.num_eqvar_heads_in
            for i in range(self.num_eqvar_heads_out):
                self.dense_eqvar.append(Dense(units=self.eqvar_units[i]*input_shape[i + offset][-1]))
                self.bn_eqvar.append(BatchNormalization(momentum=self.bn_momentum, epsilon=self.bn_epsilon))
        else:
            self.dense_eqvar = None
            self.bn_eqvar = None

    def call(self, inputs):
        invar = None
        eqvar = None
        if self.eqvar_units is not None:
            assert(isinstance(inputs,list))
            eqvar = []
            V = inputs[self.num_invar + 2*self.num_eqvar_heads_in:]
            dot_prods = DotProd(normalize=True)(inputs[self.num_invar:self.num_invar+2*self.num_eqvar_heads_in])
            invar_features = Concatenate()(inputs[0], dot_prods)
            if self.invar_units is not None:
                invar = self.dense_invar(invar_features)
            for l in range(self.num_eqvar_heads_out):
                wl = self.dense_eqvar[l](Concatenate()(invar_features))
                if self.bn_eqvar is not None:
                    wl = self.bn_eqvar[l](wl)
                if self.bn_invar is not None:
                    wl = Activation(self.activation)(wl)
                wl_shape = list(wl.shape)
                wl_shape.append(int( wl_shape[-1] / self.eqvar_units[l]))
                wl_shape[-2] = self.eqvar_units[l]
                wl = tf.reshape(wl, wl_shape)
                eqvar.append(tf.matmul(wl, V[l]))
        else:
            if self.invar_units is not None:
                invar = self.dense_invar(inputs[0])
        if invar is not None:
            if self.bn_invar is not None:
                invar = self.bn_invar(invar)
            if self.activation is not None:
                invar = Activation(self.activation)(invar)

        if eqvar is not None:
            return [invar] + eqvar
        else:
            return invar
"""
