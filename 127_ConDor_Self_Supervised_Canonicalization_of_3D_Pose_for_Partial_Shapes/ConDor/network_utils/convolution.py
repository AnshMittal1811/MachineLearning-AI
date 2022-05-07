import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Concatenate, Add
from spherical_harmonics.kernels import ZernikeKernelConv
from group_points import GroupPoints
from pooling import GridBatchSampler, GridBatchPooling
from activations import EquivariantActivationModule, L2Normalize



# ADD NORM NON LINEARITY
class EquivariantDenseLayer(tf.keras.Model):
    def __init__(self, units):
        super(EquivariantDenseLayer, self).__init__()
        assert(isinstance(units, dict))
        self.units = units
        self.dense = dict()
        for l in self.units:
            self.dense[l] = Dense(self.units[l], use_bias=(l==0))

    def call(self, inputs):
        assert(isinstance(inputs, dict))
        y = dict()
        for l in self.dense:
            y[l] = self.dense[l](inputs[l])
        return y

def eqvar_heads_fuzion(X):
    y = dict()
    for x in X:
        for l in x:
            if l not in y:
                y[l] = []
            y[l].append(x[l])
    for l in y:
        y[l] = Concatenate()(y[l])
    return y

def mask(x, mask):
    assert (isinstance(x, dict))
    mask_ = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1)
    for l in x:
        if isinstance(l, int):
            x[l] = tf.multiply(mask_, x[l])
    return x

class Mask(tf.keras.layers.Layer):
    def __init__(self, mask_key="mask"):
        super(Mask, self).__init__()
        self.mask_key = mask_key
    def build(self, input_shape):
        super(Mask, self).build(input_shape)

    def call(self, x):
        assert (isinstance(x, dict))
        mask = tf.expand_dims(tf.expand_dims(x[self.mask_key], axis=-1), axis=-1)
        for l in x:
            if isinstance(l, int):
                x[l] = tf.multiply(mask, x[l])
        return x


class EquivariantConv(tf.keras.Model):
    def __init__(self, units, d, radius):
        self.units = units
        self.d = d
        self.radius = radius
        self.zernike = ZernikeKernelConv(d=d, radius=radius)
        self.dense = EquivariantDenseLayer(units=units)
        super(EquivariantConv, self).__init__()

    def call(self, x):
        assert(isinstance(x, dict))
        return self.dense(self.zernike(x))


class ResidualEquivariantModule(tf.keras.Model):
    def __init__(self, d_max, radius, units, T_size, U_size, V_size, invar_bottleneck=None,
                 activation='relu', bn_momentum=0.99, bn_epsilon=0.01, eps=0.01, skip_after_conv=False):
        super(ResidualEquivariantModule, self).__init__()
        assert(isinstance(units, dict))
        assert ((T_size is None and U_size is None) or (T_size is not None and U_size is not None))
        if (V_size is not None):
            assert (isinstance(V_size, dict))
        else:
            V_size = dict()
        if (T_size is not None):
            assert (isinstance(T_size, dict) and isinstance(U_size, dict))
        else:
            T_size = dict()
            U_size = dict()

        self.d_max = d_max
        self.radius = radius
        self.T_size = T_size
        self.U_size = U_size
        self.V_size = V_size
        self.units = units
        self.invar_bottleneck_dense = None
        self.skip_after_conv = skip_after_conv
        self.eps = eps

        self.normalization = L2Normalize(eps=eps, axis=-2)
        if 0 not in units:
            units_invar = None
            units_eqvar = units.values()
        else:
            units_invar = units[0]
            units_eqvar = units.values()
            units_eqvar = units_eqvar[1:]

        self.activation_layer = EquivariantActivationModule(units_invar=units_invar, units_eqvar=units_eqvar,
                                                            size_eqvar_basis=V_size, activation=activation,
                                                            bn_momentum=bn_momentum, bn_epsilon=bn_epsilon,
                                                            eps=eps)
        if invar_bottleneck is not None:
            self.invar_bottleneck_dense = Dense(units=invar_bottleneck, use_bias=False)
        self.skip_connections = dict()
        self.T_dense = dict()
        for t in T_size:
            self.T_dense[t] = Dense(units=T_size[t], use_bias=False)
        self.U_dense = dict()
        for u in U_size:
            self.U_dense[u] = Dense(units=U_size[u], use_bias=False)

        self.zernike = ZernikeKernelConv(d=d_max, radius=radius)

        for l in V_size:
            self.V_dense[l] = Dense(units=V_size[l], use_bias=False)
            self.skip_connections[l] = Dense(units=V_size[l], use_bias=False)

        if 0 in self.units:
            self.skip_connections[0] = Dense(units=V_size[0], use_bias=False)

    def call(self, inputs):
        """
        :param inputs: {l_0:x_0, ..., l_k:x_k, "points":pts, "patches idx":idx, "mask":mask, "patches size":patches_size}
        :return: {J_0:y_0, ... , J_m:y_m}
        """
        assert(isinstance(inputs, dict))
        if "source mask" in inputs:
            inputs = Mask("source mask")(inputs)
        y = self.zernike(inputs)

        if 0 in y:
            if self.invar_bottleneck_dense is not None:
                y[0] = self.invar_bottleneck_dense(y[0])

        y_before_activation = y
        if self.T_size is not None:
            T = dict()
            U = dict()
            for l in self.T_size:
                T[l] = self.T_dense[l](y[l])
                U[l] = self.T_dense[l](y[l])
            T_values = T.values()
            U_values = U.values()
        else:
            T_values = []
            U_values = []
        if self.V_size is not None:
            V = dict()
            for l in self.V_size:
                V[l] = self.V_dense[l](y[l])
            V = self.normalization(V)
            V_values = V.values()
        else:
            V_values = []
        if 0 in y:
            y = {0:y[0]}
        y = y + T_values + U_values + V_values
        y = self.activation_layer(y)

        for l in self.skip_connections:
            if self.skip_after_conv:
                y[l] = Add()([y[l], self.skip_connections[l](y_before_activation[l])])
            else:
                y[l] = Add()([y[l], self.skip_connections[l](inputs[l])])

        if "source mask" in inputs:
            y["target mask"] = inputs["target mask"]
            y = Mask("target mask")(y)
            y.pop("target mask")
        return y



