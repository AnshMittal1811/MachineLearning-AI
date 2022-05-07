from sympy import *
import numpy as np
import tensorflow as tf
from spherical_harmonics.clebsch_gordan_decomposition import tf_clebsch_gordan_decomposition
from spherical_harmonics.tf_spherical_harmonics import normalized_real_sh
import h5py
from network_utils.group_points import GroupPoints

def associated_legendre_polynomial(l, m, z, r2):
    P = 0
    if l < m:
        return P
    for k in range(int((l-m)/2)+1):
        pk = (-1.)**k * (2.)**(-l) * binomial(l, k) * binomial(2*l-2*k, l)
        pk *= (factorial(l-2*k) / factorial(l-2*k-m)) * r2**k * z**(l-2*k-m)
        P += pk
    P *= np.sqrt(float(factorial(l-m)/factorial(l+m)))
    return P

def A(m, x, y):
    a = 0
    for p in range(m+1):
        a += binomial(m, p) * x**p * y**(m-p) * cos((m-p)*(pi/2.))
    return a

def B(m, x, y):
    b = 0
    for p in range(m+1):
        b += binomial(m, p) * x**p * y**(m-p) * sin((m-p)*(pi/2.))
    return b
"""
computes the unnormalized real spherical harmonic Y_{lm} as a polynomial
of the euclidean coordinates x, y, z
"""
def real_spherical_harmonic(l, m, x, y, z):
    K = np.sqrt((2*l+1)/(2*np.pi))
    r2 = x**2 + y**2 + z**2
    if m > 0:
        Ylm = K * associated_legendre_polynomial(l, m, z, r2) * A(m, x, y)
    elif m < 0:
        Ylm = K * associated_legendre_polynomial(l, -m, z, r2) * B(-m, x, y)
    else:
        K = np.sqrt((2 * l + 1) / (4 * np.pi))
        Ylm = K * associated_legendre_polynomial(l, 0, z, r2)
    return Ylm

def binom(n, k):
    if k == 0.:
        return 1.
    return gamma(n + 1) / (gamma(n-k+1)*gamma(k+1))

"""
computes radial Zernike polynomials (divided by r^{2l})
"""
def zernike_polynomial_radial(n, l, D, r2):
    if (l > n):
        return 0
    if ((n-l) % 2 != 0):
        return 0
    R = 0
    for s in range(int((n-l) / 2) + 1):
        R += (-1)**s * binom((n-l)/2, s)*binom(s-1+(n+l+D)/2., (n-l)/2)*r2**s
    R *= (-1)**((n-l)/2)*np.sqrt(2*n+D)
    return R

"""
computes the 3D Zernike polynomials.
"""
def zernike_kernel_3D(n, l, m, x, y, z):
    r2 = x**2 + y**2 + z**2
    return zernike_polynomial_radial(n, l, 3, r2)*real_spherical_harmonic(l, m, x, y, z)
    # return real_spherical_harmonic(l, m, x, y, z)

"""
computes the monomial basis in x, y, z up to degree d
"""
def monomial_basis_3D(d):
    monoms_basis = []
    for I in range((d + 1) ** 3):
        i = I % (d + 1)
        a = int((I - i) / (d + 1))
        j = a % (d + 1)
        k = int((a - j) / (d + 1))
        if (i + j + k <= d):
            monoms_basis.append((i, j, k))

    monoms_basis = list(set(monoms_basis))
    monoms_basis = sorted(monoms_basis)
    return monoms_basis

def tf_monomial_basis_3D_idx(d):
    m = monomial_basis_3D(d)
    idx = np.zeros((len(m), 3), dtype=np.int32)
    for i in range(len(m)):
        for j in range(3):
            idx[i, j] = m[i][j]
    return tf.convert_to_tensor(idx, dtype=tf.int64)

"""
computes the coefficients of a list of polynomials in the monomial basis
"""
def np_monomial_basis_coeffs(polynomials, monoms_basis):
    n_ = len(monoms_basis)
    m_ = len(polynomials)
    M = np.zeros((m_, n_))
    for i in range(m_):
        for j in range(n_):
            M[i, j] = polynomials[i].coeff_monomial(monoms_basis[j])
    return M

def tf_monomial_basis_coeffs(polynomials, monoms_basis, dtype=tf.float32):
    return tf.convert_to_tensor(np_monomial_basis_coeffs(polynomials, monoms_basis), dtype=dtype)

"""
computes 1d grid kernels
"""
# Just stack filters for multiscale support
def tf_gaussian_polynomial_1d_grid_kernel(gaussian_scale, d, size, per_bin=10, kernel_scale=1., dtype=tf.float32):
    gaussian_scale = gaussian_scale / (kernel_scale**2)
    kernel = (kernel_scale/(size*per_bin))*tf.range(-size*per_bin, size*per_bin+1, dtype=tf.float32)
    kernel = tf.expand_dims(kernel, axis=0)
    g = tf.exp(-gaussian_scale*tf.expand_dims(tf.multiply(kernel, kernel), axis=1))
    pows = tf.expand_dims(tf.range(d+1, dtype=tf.float32), axis=0)
    kernel = tf.multiply(g, tf.pow(kernel, pows))
    kernel = tf.reshape(kernel, (2*size+1, per_bin, -1))
    kernel = tf.reduce_mean(kernel, axis=1, keepdims=False)
    # kernel = tf.transpose(kernel, (1, 0))
    return tf.cast(kernel, dtype=dtype)


"""
computes the coefficients of the spherical harmonics in the monomial basis
"""
def spherical_harmonics_3D_monomial_basis(l, monoms_basis):
    x, y, z = symbols("x y z")
    n_ = len(monoms_basis)
    M = np.zeros((2*l+1, n_))
    for m in range(2*l+1):
        Y = real_spherical_harmonic(l, m-l, x, y, z)
        Y = expand(Y)
        Y = poly(Y, x, y, z)
        for i in range(n_):
            M[m, i] = N(Y.coeff_monomial(monoms_basis[i]))
    return M

"""
computes the coefficients of the Zernike polynomials in the monomial basis
"""
def zernike_kernel_3D_monomial_basis(n, l, monoms_basis):
    x, y, z = symbols("x y z")
    n_ = len(monoms_basis)
    M = np.zeros((2*l+1, n_))
    for m in range(2*l+1):
        Z = zernike_kernel_3D(n, l, m-l, x, y, z)
        Z = expand(Z)
        Z = poly(Z, x, y, z)
        for i in range(n_):
            M[m, i] = N(Z.coeff_monomial(monoms_basis[i]))
    return M


"""
computes the matrix of an offset in the monomial basis (up to degree d)
(m_1(x-a), ..., m_k(x-a)) = A(a).(m_1(x), ..., m_k(x))
"""
def np_monom_basis_offset(d):
    monoms_basis = monomial_basis_3D(d)
    n = len(monoms_basis)
    idx = np.full(fill_value=-1, shape=(n, n), dtype=np.int32)
    coeffs = np.zeros(shape=(n, n))

    for i in range(n):
        pi = monoms_basis[i][0]
        qi = monoms_basis[i][1]
        ri = monoms_basis[i][2]
        for j in range(n):
            pj = monoms_basis[j][0]
            qj = monoms_basis[j][1]
            rj = monoms_basis[j][2]
            if (pj >= pi) and (qj >= qi) and (rj >= ri):
                idx[j, i] = monoms_basis.index((pj-pi, qj-qi, rj-ri))
                coeffs[j, i] = binomial(pj, pi)*binomial(qj, qi)*binomial(rj, ri)*((-1.)**(pj-pi+qj-qi+rj-ri))
    return coeffs, idx

def tf_monom_basis_offset(d):
    coeffs, idx = np_monom_basis_offset(d)
    coeffs = tf.convert_to_tensor(coeffs, dtype=tf.float32)
    idx = tf.convert_to_tensor(idx, dtype=tf.int64)
    return coeffs, idx

"""
computes the 3D zernike basis up to degree d
"""
def np_zernike_kernel_basis(d):
    monoms_basis = monomial_basis_3D(d)
    Z = []
    for l in range(d+1):
        Zl = []
        # for n in range(min(2*d - l + 1, l + 1)):
        #    if (n - l) % 2 == 0:
        for n in range(l, d+1):
            if (n - l) % 2 == 0 and d >= n:
                Zl.append(zernike_kernel_3D_monomial_basis(n, l, monoms_basis))
        Z.append(np.stack(Zl, axis=0))
    return Z

def tf_zernike_kernel_basis(d, stack_axis=1):
    monoms_basis = monomial_basis_3D(d)
    Z = []
    for l in range(d+1):
        Zl = []
        for n in range(l, d+1):
            if (n - l) % 2 == 0 and d >= n:
                Zl.append(zernike_kernel_3D_monomial_basis(n, l, monoms_basis))
        Z.append(tf.convert_to_tensor(np.stack(Zl, axis=stack_axis), dtype=tf.float32))
    return Z


"""
computes the 3D spherical harmonics basis up to degree l_max
"""
def tf_spherical_harmonics_basis(l_max, concat=False):
    monoms_basis = monomial_basis_3D(l_max)
    Y = []
    for l in range(l_max+1):
        Yl = spherical_harmonics_3D_monomial_basis(l, monoms_basis)
        Y.append(tf.convert_to_tensor(Yl, dtype=tf.float32))
    if concat:
        Y = tf.concat(Y, axis=0)
    return Y

def np_zernike_kernel(d, n, l):
    monoms_basis = monomial_basis_3D(d)
    assert (n >= l and (n - l) % 2 == 0)
    return zernike_kernel_3D_monomial_basis(n, l, monoms_basis)

def tf_zernike_kernels(kernels):
    x, y, z = symbols("x y z")
    monoms_basis = []
    polynoms = dict()
    idx_size = dict()

    for i in range(len(kernels)):
        n = kernels[i][0]
        l = kernels[i][1]
        if l not in polynoms:
            polynoms[l] = []
            idx_size[l] = 0
        Znl = []
        for m in range(2 * l + 1):
            Z = zernike_kernel_3D(n, l, m - l, x, y, z)
            Z = poly(Z, x, y, z)
            Znl.append(Z)
            Z_monoms = Z.monoms()
            idx_size[l] = max(idx_size[l], len(Z_monoms))
            monoms_basis += Z.monoms()
        polynoms[l].append(Znl)

    monoms_basis = sorted(list(set(monoms_basis)))
    idx = dict()
    coeffs = dict()
    for l in polynoms:
        idx[l] = []
        coeffs[l] = []
        for Z in polynoms[l]:
            idx_Z = -np.ones((2*l+1, idx_size[l]))
            coeffs_Z = np.zeros((2 * l + 1, idx_size[l]))
            for m in range(len(Z)):
                Zm_monoms = Z[m].monoms()
                for i in range(len(Zm_monoms)):
                    idx_Z[m, i] = monoms_basis.index(Zm_monoms[i])
                    coeffs_Z[m, i] = Z[m].coeff_monomial(Zm_monoms[i])
    for l in polynoms:
        idx[l] = tf.convert_to_tensor(np.stack(idx[l], axis=1), dtype=tf.int32)
        coeffs[l] = tf.convert_to_tensor(np.stack(coeffs[l], axis=1), dtype=tf.float32)

    monoms_basis_idx = -np.ones((3, len(monoms_basis)))
    for j in range(len(monoms_basis)):
        for i in range(3):
            monoms_basis_idx[i, j] = monoms_basis[j][i]
    monoms_basis_idx = tf.convert_to_tensor(monoms_basis_idx, tf.int32)
    return coeffs, idx, monoms_basis_idx

def tf_spherical_harmonics(sph):
    x, y, z = symbols("x y z")
    monoms_basis = []
    polynoms = dict()
    idx_size = dict()

    for i in range(len(sph)):
        l = sph[i]
        if l not in polynoms:
            polynoms[l] = []
            idx_size[l] = 0
        Yl = []
        for m in range(2 * l + 1):
            Ylm = real_spherical_harmonic(l, m-l, x, y, z)
            Ylm = expand(Ylm)
            Ylm = poly(Ylm, x, y, z)
            Yl.append(Ylm)
            Ylm_monoms = Ylm.monoms()
            idx_size[l] = max(idx_size[l], len(Ylm_monoms))
            monoms_basis += Ylm.monoms()
        polynoms[l].append(Yl)

    monoms_basis = sorted(list(set(monoms_basis)))
    idx = dict()
    coeffs = dict()
    for l in polynoms:
        idx[l] = -np.ones((2*l+1, idx_size[l]))
        coeffs[l] = np.zeros((2 * l + 1, idx_size[l]))
        for Z in polynoms[l]:
            for m in range(len(Z)):
                Zm_monoms = Z[m].monoms()
                for i in range(len(Zm_monoms)):
                    idx[l][m, i] = monoms_basis.index(Zm_monoms[i])
                    coeffs[l][m, i] = Z[m].coeff_monomial(Zm_monoms[i])
    for l in polynoms:
        idx[l] = tf.convert_to_tensor(idx[l], dtype=tf.int32)
        coeffs[l] = tf.convert_to_tensor(coeffs[l], dtype=tf.float32)

    monoms_basis_idx = -np.ones((3, len(monoms_basis)))
    for j in range(len(monoms_basis)):
        for i in range(3):
            monoms_basis_idx[i, j] = monoms_basis[j][i]
    monoms_basis_idx = tf.convert_to_tensor(monoms_basis_idx, tf.int32)
    return coeffs, idx, monoms_basis_idx



"""
evaluate monomial basis up to degree d
"""
def tf_eval_monom_basis(x, d, idx=None):
    batch_size = x.shape[0]
    num_points = x.shape[1]

    if idx is None:
        idx = tf_monomial_basis_3D_idx(d)
    y = []
    for i in range(3):
        pows = tf.reshape(tf.range(d+1, dtype=tf.float32), (1, 1, d+1))
        yi = tf.pow(tf.expand_dims(x[..., i], axis=-1), pows)
        y.append(tf.gather(yi, idx[..., i], axis=-1))
    y = tf.stack(y, axis=-1)
    y = tf.reduce_prod(y, axis=-1, keepdims=False)
    return y

def tf_eval_monoms(x, monoms_idx):
    monoms_idx = tf.reshape(tf.cast(monoms_idx, tf.float32), [1]*(len(list(x.shape))-1) + list(monoms_idx.shape))
    x = tf.expand_dims(x, axis=-1)
    y = tf.pow(x, monoms_idx)
    return tf.reduce_prod(y, axis=-2, keepdims=False)

"""
compute the offset of the monomial basis for all points
"""
def compute_monomial_basis_offset(x, offset_monoms, idx, coeffs):
    offset_matrix = tf.gather(offset_monoms, idx, axis=-1)
    offset_matrix = tf.multiply(offset_matrix, coeffs)
    # return tf.matmul(offset_matrix, x)
    return tf.einsum('...ij,...mjk->...mik', offset_matrix, x)

def mask_mult(mask, x):
    d = len(x.shape)
    k = len(mask.shape)
    mask = tf.reshape(mask, list(mask.shape) + [1]*(d-k))
    return tf.multiply(mask, x)

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

class GaussianShKernel(tf.keras.layers.Layer):
    def __init__(self, l, nr):
        super(GaussianShKernel, self).__init__()
        self.l = l
        self.nr = nr
        coeffs, idx, monoms_idx = tf_spherical_harmonics(l)
        self.sph_coeffs = coeffs
        self.sph_idx = idx
        self.monoms_idx = tf.reshape(tf.cast(monoms_idx, tf.float32), [1, 1, 1] + list(monoms_idx.shape))

    def build(self, input_shape):
        super(GaussianShKernel, self).build(input_shape)

    def call(self, x):
        patches = tf.expand_dims(x["patches"], axis=-1)
        ball_querry_idx = x["patches idx"]
        num_points_per_ball = tf.reduce_mean(x["patches size"], axis=1, keepdims=True)
        num_points_per_ball = tf.reshape(num_points_per_ball, list(num_points_per_ball.shape) + [1, 1, 1])
        source_mask = None
        if "source mask" in x:
            source_mask = tf.gather_nd(x["source mask"], ball_querry_idx)
            source_mask = tf.reshape(source_mask, list(source_mask.shape) + [1, 1])

        # monoms = tf.reduce_prod(tf.pow(patches, self.monoms_idx), axis=-2, keepdims=False)
        kernels = dict()
        g = tf.expand_dims(gaussian_shells(x["patches dist"], self.nr), axis=-2)
        Y = normalized_real_sh(l_max=3, X=x["patches"], r=None)

        for l in range(4):
            key = "kernel_" + str(l)
            # kernels[key] = tf.einsum('mi,bvpmi->bvpm', self.sph_coeffs[l], tf.gather(monoms, self.sph_idx[l], axis=-1))
            # kernels[key] = tf.multiply(g, tf.expand_dims(kernels[key], axis=-1))
            kernels[key] = tf.multiply(g, tf.expand_dims(Y[l], axis=-1))
            if source_mask is not None:
                kernels[key] = tf.multiply(source_mask, kernels[key])
            kernels[key] = tf.divide(kernels[key], num_points_per_ball)
        return kernels



"""
class ZernikeKernel(tf.keras.layers.Layer):
    def __init__(self, kernels):
        super(ZernikeKernel, self).__init__()
        self.kernels = kernels
        coeffs, idx, monoms_idx = tf_zernike_kernels(kernels)
        self.kernels_coeffs = coeffs
        self.kernels_idx = idx
        self.monoms_idx = tf.reshape(tf.cast(monoms_idx, tf.float32), [1, 1, 1] + list(monoms_idx.shape))

    def build(self, input_shape):
        super(ZernikeKernel, self).build(input_shape)

    def call(self, x):
        patches = tf.expand_dims(x["patches"], axis=-1)
        ball_querry_idx = x["patches idx"]
        num_points_per_ball = tf.reduce_mean(x["patches size"], axis=1, keepdims=True)
        num_points_per_ball = tf.reshape(num_points_per_ball, list(num_points_per_ball.shape) + [1, 1])
        source_mask = None
        if "source mask" in x:
            source_mask = tf.gather_nd(x["source mask"], ball_querry_idx)
            source_mask = tf.reshape(source_mask, list(source_mask.shape) + [1, 1])

        monoms = tf.reduce_prod(tf.pow(patches, self.monoms_idx), axis=-2, keepdims=False)
        kernels = dict()
        for l in self.kernels_coeffs:
            key = str(l)
            kernels[key] = tf.einsum('mni,bvpi->bvpmn', self.kernels_coeffs[l],
                                   tf.gather(monoms, self.kernels_idx[l], axis=-1))
            if source_mask is not None:
                kernels[key] = tf.multiply(source_mask, kernels[key])
            kernels[key] = tf.divide(kernels[key], num_points_per_ball)
        return kernels
"""
def pack_kernels_and_signals(kernels, signals):
    y = dict()
    for k in kernels:
        if k.isnumeric():
            y["kernel_" + k] = kernels[k]
        else:
            y[k] = kernels[k]
    for l in signals:
        if l.isnumeric():
            y["signal_" + l] = signals[l]
        else:
            y[l] = signals[l]
    return y

class EquivariantConv(tf.keras.layers.Layer):
    def __init__(self, l_max):
        super(EquivariantConv, self).__init__()
        self.l_max = l_max
        self.Q = tf_clebsch_gordan_decomposition(l_max=l_max, sparse=False, output_type='dict')

    def build(self, input_shape):
        super(EquivariantConv, self).build(input_shape)

    def call(self, x):
        assert(isinstance(x, dict))

        source_mask = None
        if "source mask" in x:
            source_mask = tf.reshape(x["source mask"], list(x["source mask"]) + [1, 1])
        target_mask = None
        if "target mask" in x:
            target_mask = tf.reshape(x["target mask"], list(x["target mask"]) + [1, 1])

        kernels = dict()
        signal = dict()
        for key in x:
            k = key[:6]
            l = key[7:]
            if k == "kernel" and l.isnumeric():
                kernels[int(l)] = x[key]
            if k == "signal" and l.isnumeric():
                signal[int(l)] = x[key]
                if source_mask is not None:
                    signal[int(l)] = tf.multiply(source_mask, signal[int(l)])

        y_cg = []
        y = {k: [] for k in kernels}

        for l in signal:
            for k in kernels:
                sl = tf.gather_nd(signal[l], x["patches idx"])
                ylk = tf.einsum('bvpkn,bvpmc->bvmknc', kernels[k], sl)
                s = list(ylk.shape)
                s.pop()
                s[-1] = -1
                ylk = tf.reshape(ylk, shape=s)
                if(l > 0):
                    y_cg.append(ylk)
                else:
                    y[k].append(ylk[:, :, 0, :, :])

        if len(y_cg) > 0:
            y_cg = self.Q.decompose(y_cg)
        for J in y_cg:
            if J not in y:
                y[J] = []
            y[J].append(y_cg[J])
        for J in y:
            y[J] = tf.concat(y[J], axis=-1)
            if target_mask is not None:
                y[J] = tf.multiply(target_mask, y[J])
        return y

def shapes(x):
    if isinstance(x, dict):
        y = dict()
        for k in x:
            if isinstance(x[k], list):
                L = []
                for xij in x[k]:
                    L.append(xij.shape)
                y[k] = L
            else:
                y[k] = x[k].shape
    if isinstance(x, list):
        y = []
        for xi in x:
            if isinstance(xi, list):
                L = []
                for xij in xi:
                    L.append(xij.shape)
                y.append(L)
            else:
                y.append(xi.shape)
    return y

class ZernikeKernelConv(tf.keras.layers.Layer):
    def __init__(self, d, radius, l_max_out=None, output_type='dict'):
        super(ZernikeKernelConv, self).__init__()
        self.d = d
        self.Z = tf_zernike_kernel_basis(d)
        self.offset_coeffs, self.offset_idx = tf_monom_basis_offset(d)
        self.radius = radius
        self.monoms_idx = tf_monomial_basis_3D_idx(d)
        self.output_type = output_type
        self.l_max_out = l_max_out
        self.Q = tf_clebsch_gordan_decomposition(l_max=d, sparse=False, output_type=output_type, l_max_out=l_max_out)

    def build(self, input_shape):
        super(ZernikeKernelConv, self).build(input_shape)

    def call(self, x):
        """
        :param inputs: {l_0:x_0, ..., l_k:x_k, "points":pts, "patches idx":idx, "mask":mask, "patches size":patches_size}
        :return: {J_0:y_0, ... , J_m:y_m}
        """
        # assert(isinstance(x, list))
        assert(isinstance(x, dict))
        int_keys = []
        for key in x:
            if isinstance(key, str):
                if key.isnumeric():
                    int_keys.append(key)

        if "target points" in x:
            # source_points = (1./self.radius)*x[-4]
            # target_points = (1./self.radius)*x[-3]
            # source_points = (1. / self.radius) * x["source points"]
            # target_points = (1. / self.radius) * x["target points"]
            source_points = tf.divide(x["source points"], x["radius"])
            target_points = tf.divide(x["target points"], x["radius"])
            source_monoms = tf_eval_monom_basis(source_points, self.d, idx=self.monoms_idx)
            target_monoms = tf_eval_monom_basis(target_points, self.d, idx=self.monoms_idx)
            # y = x[:-3]
        else:
            # source_points = (1. / self.radius) * x["source points"]
            source_points = tf.divide(x["points"], x["radius"])
            target_points = source_points
            source_monoms = tf_eval_monom_basis(source_points, self.d, idx=self.monoms_idx)
            target_monoms = source_monoms
            # y = x[:-3]
        batch_size = source_points.shape[0]
        num_points_source = source_points.shape[1]
        num_points_target = target_points.shape[1]
        ball_querry_idx = x["patches idx"]
        # num_points_per_ball = tf.scatter_nd(ball_querry_idx, tf.ones(target_points.shape), shape=target_monoms.shape)
        num_points_per_ball = x["patches size"]
        num_points_per_ball = tf.reduce_mean(num_points_per_ball, axis=1, keepdims=True)
        num_points_per_ball = tf.reshape(num_points_per_ball, (batch_size, 1, 1, 1, 1))
        # source_monoms = tf.reshape(source_monoms, (batch_size, num_points_source, 1, source_monoms.shape[-1], 1))
        source_monoms = tf.gather_nd(source_monoms, ball_querry_idx)
        source_monoms = tf.where(tf.expand_dims(x["patches dist source"], axis=-1) <= 1., source_monoms, 0.)

        source_mask = None
        if "source mask" in x:
            source_mask = tf.reshape(x["source mask"], (batch_size, num_points_source, 1, 1))
        target_mask = None
        if "target mask" in x:
            target_mask = tf.reshape(x["target mask"], (batch_size, num_points_target, 1, 1))

        y_cg = []

        y = {str(k): [] for k in range(len(self.Z))}
        for l_ in int_keys:
            l = int(l_)
            yl = x[l_]
            # if yl == 0:
            #    yl = tf.expand_dims(yl, axis=-2)
            if source_mask is not None:
                yl = tf.multiply(source_mask, yl)
            # yl = tf.multiply(source_monoms, tf.expand_dims(yl, axis=-2))
            yl = tf.gather_nd(yl, ball_querry_idx)
            yl = tf.einsum('bvpk,bvpmc->bvmkc', source_monoms, yl)

            yl = tf.divide(yl, num_points_per_ball)
            yl = compute_monomial_basis_offset(yl, target_monoms, self.offset_idx, self.offset_coeffs)
            for k in range(len(self.Z)):
                ylk = tf.einsum('mnj,...jc->...mnc', self.Z[k], yl)
                s = list(ylk.shape)
                s.pop()
                s[-1] = -1
                ylk = tf.reshape(ylk, shape=s)
                if(l > 0 and k > 0):
                    y_cg.append(ylk)
                elif(l == 0):
                    y[str(k)].append(ylk[:, :, 0, :, :])
                else:
                    y[str(l)].append(ylk[:, :, :, 0, :])


        if len(y_cg) > 0:
            y_cg = self.Q.decompose(y_cg)

        for J in y_cg:
            if J not in y:
                y[J] = []
            y[J].append(y_cg[J])
        for J in y:

            y[J] = tf.concat(y[J], axis=-1)
            if target_mask is not None:
                y[J] = tf.multiply(target_mask, y[J])
        return y


class ZernikeGaussianKernels(tf.keras.layers.Layer):
    def __init__(self, d, gaussian_scale, normalization='L2'):
        super(ZernikeGaussianKernels, self).__init__()
        self.d = d
        self.monoms_idx = tf_monomial_basis_3D_idx(d)
        self.gaussian_scale = gaussian_scale
        self.normalization = normalization
        Z = tf_zernike_kernel_basis(d)
        self.zernike_split_size = []
        self.split_size = []
        self.Z = []
        self.zernike_idx = []
        k = 0
        for l in range(len(Z)):
            for j in range(Z[l].shape[1]):
                # self.split_size.append(Z[l].shape[0])
                self.zernike_idx += [k]*(2*l+1)
                k += 1
            self.zernike_split_size.append(Z[l].shape[0] * Z[l].shape[1])
            self.Z.append(tf.reshape(Z[l], (-1, Z[l].shape[-1])))
        self.Z = tf.concat(self.Z, axis=0)
        self.zernike_idx = tf.convert_to_tensor(np.array(self.zernike_idx), dtype=tf.int32)
    def build(self, input_shape):
        super(ZernikeGaussianKernels, self).build(input_shape)

    def call(self, x):
        monoms_patches = tf_eval_monom_basis(x["patches"], self.d, idx=self.monoms_idx)
        # zernike_patches = tf.einsum('zm,bvpm'self.Z)
        # zernike_patches = tf.matmul(self.Z, monoms_patches)

        zernike_patches = tf.einsum('ij,bvpj->bvpi', self.Z, monoms_patches)


        patches_dist = tf.expand_dims(x["patches dist"], axis=-1)
        g = tf.exp(-self.gaussian_scale*tf.multiply(patches_dist, patches_dist))
        zernike_patches = tf.multiply(g, zernike_patches)
        zernike_patches = tf.where(patches_dist <= 1., zernike_patches, 0.)

        if self.normalization == "L1":
            # L1 regularization
            l1_norm = tf.reduce_sum(tf.abs(zernike_patches), axis=2, keepdims=True)
            l1_norm = tf.reduce_mean(l1_norm, axis=1, keepdims=True)
            l1_norm = tf.split(l1_norm, num_or_size_splits=self.zernike_split_size, axis=-1)
            Z = []
            for l in range(len(l1_norm)):
                l1l = tf.reshape(l1_norm[l], (l1_norm[l].shape[0], l1_norm[l].shape[1], 2*l+1, -1))
                Z.append(tf.reduce_mean(l1l, axis=-2, keepdims=True))
            l1_norm = tf.maximum(tf.concat(Z, axis=-1), 1e-8)
            l1_norm = tf.gather(l1_norm, self.zernike_idx, axis=-1)
            zernike_patches = tf.divide(zernike_patches, l1_norm)
            # return {"kernels": zernike_patches}
        elif self.normalization == "L2":
            l2_norm = tf.reduce_sum(tf.multiply(zernike_patches, zernike_patches), axis=2, keepdims=True)
            l2_norm = tf.sqrt(l2_norm)
            l2_norm = tf.reduce_mean(l2_norm, axis=1, keepdims=True)
            l2_norm = tf.split(l2_norm, num_or_size_splits=self.zernike_split_size, axis=-1)
            Z = []
            for l in range(len(l2_norm)):
                l2l = tf.reshape(l2_norm[l], (l2_norm[l].shape[0], l2_norm[l].shape[1], 2*l+1, -1))
                Z.append(tf.reduce_mean(l2l, axis=-2, keepdims=True))
            l2_norm = tf.maximum(tf.concat(l2_norm, axis=-1), 1e-8)
            l2_norm = tf.gather(l2_norm, self.zernike_idx, axis=-1)
            zernike_patches = tf.divide(zernike_patches, l2_norm)




        return zernike_patches



class ZernikeGaussianKernelConv(tf.keras.layers.Layer):
    def __init__(self, d, l_max_out=None, output_type='dict'):
        super(ZernikeGaussianKernelConv, self).__init__()
        self.d = d
        Z = tf_zernike_kernel_basis(d)
        self.zernike_split_size = []
        self.Z = []
        for l in range(len(Z)):
            self.zernike_split_size.append(Z[l].shape[0] * Z[l].shape[1])
            self.Z.append(tf.reshape(Z[l], (-1, Z[l].shape[-1])))
        self.Z = tf.concat(self.Z, axis=0)

        # self.offset_coeffs, self.offset_idx = tf_monom_basis_offset(d)
        # self.monoms_idx = tf_monomial_basis_3D_idx(d)
        self.output_type = output_type
        self.l_max_out = l_max_out
        self.Q = tf_clebsch_gordan_decomposition(l_max=d, sparse=False, output_type=output_type, l_max_out=l_max_out)

    def build(self, input_shape):
        super(ZernikeGaussianKernelConv, self).build(input_shape)

    def call(self, x):
        assert (isinstance(x, dict))
        signal = []
        features_type = []
        channels_split_size = []
        for l in x:
            if l.isnumeric():
                features_type.append(int(l))
                # channels_split_size .append(x[l].shape[-2]*x[l].shape[-1])
                # signal.append(tf.reshape(x[l], (x[l].shape[0], -1)))
                channels_split_size.append(x[l].shape[-2] * x[l].shape[-1])
                signal.append(tf.reshape(x[l], (x[l].shape[0], x[l].shape[1], -1)))

        signal = tf.concat(signal, axis=-1)
        signal = tf.gather_nd(signal, x["patches idx"])
        batch_size = signal.shape[0]
        num_points_target = signal.shape[1]

        # signal = tf.expand_dims(signal, axis=1)

        y = tf.einsum('bvpz,bvpc->bvzc', x["kernels"], signal)
        # split y
        y_ = tf.split(y, num_or_size_splits=channels_split_size, axis=-1)
        y = {str(j): [] for j in range(self.d + 1)}
        y_cg = []
        for i in range(len(channels_split_size)):
            l = features_type[i]
            # yi = tf.reshape(y[i], (self._build_input_shape[str(l)][0], -1, self._build_input_shape[str(l)][-1]))
            yi = tf.reshape(y_[i], (batch_size, num_points_target, -1, 2 * l + 1, x[str(l)].shape[-1]))
            yi = tf.transpose(yi, (0, 1, 3, 2, 4))
            yi = tf.split(yi, num_or_size_splits=self.zernike_split_size, axis=-2)
            for j in range(len(self.zernike_split_size)):
                # yij = tf.transpose(yi[j], (0, 2, 1, 3))
                yij = tf.reshape(yi[j], (batch_size, num_points_target, 2 * l + 1, 2 * j + 1, -1))
                if l == 0:
                    y[str(j)].append(yij[:, :, 0, :, :])
                elif j == 0:
                    y[str(l)].append(yij[:, :, :, 0, :])
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


class SphericalHarmonicsGaussianKernels(tf.keras.layers.Layer):
    def __init__(self, l_max, gaussian_scale, num_shells, transpose=False, bound=True):
        super(SphericalHarmonicsGaussianKernels, self).__init__()
        self.l_max = l_max
        self.monoms_idx = tf_monomial_basis_3D_idx(l_max)
        self.gaussian_scale = gaussian_scale
        self.num_shells = num_shells
        self.transpose = True
        self.Y = tf_spherical_harmonics_basis(l_max, concat=True)
        self.split_size = []
        self.sh_idx = []
        self.bound = bound
        for l in range(l_max + 1):
            self.split_size.append(2*l+1)
            self.sh_idx += [l]*(2*l+1)
        self.sh_idx = tf.convert_to_tensor(np.array(self.sh_idx), dtype=tf.int32)

    def build(self, input_shape):
        super(SphericalHarmonicsGaussianKernels, self).build(input_shape)

    def call(self, x):
        if "patches dist" in x:
            patches_dist = tf.expand_dims(x["patches dist"], axis=-1)
        else:
            patches_dist = tf.norm(x["patches"], axis=-1, keepdims=True)
        normalized_patches = tf.divide(x["patches"], tf.maximum(patches_dist, 0.000001))
        if self.transpose:
            normalized_patches = -normalized_patches
        monoms_patches = tf_eval_monom_basis(normalized_patches, self.l_max, idx=self.monoms_idx)
        sh_patches = tf.einsum('ij,bvpj->bvpi', self.Y, monoms_patches)
        shells_rad = tf.range(self.num_shells, dtype=tf.float32) / (self.num_shells-1)

        shells_rad = tf.reshape(shells_rad, (1, 1, 1, -1))
        shells = tf.subtract(patches_dist, shells_rad)
        shells = tf.exp(-self.gaussian_scale*tf.multiply(shells, shells))
        shells_sum = tf.reduce_sum(shells, axis=-1, keepdims=True)
        shells = tf.divide(shells, tf.maximum(shells_sum, 0.000001))

        """
        # normalize shells
        shells_mean = tf.reduce_sum(shells, axis=2, keepdims=True)
        shells_mean = tf.reduce_mean(shells_mean, axis=1, keepdims=True)
        shells = tf.divide(shells, shells_mean)
        """

        shells = tf.expand_dims(shells, axis=-2)
        if self.bound:
            shells = tf.where(tf.expand_dims(patches_dist, axis=-1) <= 1., shells, 0.)


        # normalize shells

        """
        shells_mean = tf.reduce_sum(shells, axis=2, keepdims=True)
        shells_mean = tf.reduce_mean(shells_mean, axis=[1, -1], keepdims=True)
        shells = tf.divide(shells, shells_mean)
        """


        sh_patches = tf.expand_dims(sh_patches, axis=-1)
        sh_patches = tf.multiply(shells, sh_patches)


        """
        # sh_patches = tf.where(tf.expand_dims(patches_dist, axis=-1) <= 1., sh_patches, 0.)
        g = tf.reshape(sh_patches[:, :, :, 0, 0], (sh_patches.shape[0], sh_patches.shape[1], sh_patches.shape[2], 1, 1))
        # g = tf.maximum(tf.reduce_sum(g, axis=2, keepdims=True), 0.000001)
        g = tf.maximum(tf.reduce_mean(tf.reduce_sum(g, axis=2, keepdims=True), axis=1, keepdims=True), 0.000001)
        sh_patches = tf.divide(sh_patches, g)
        """


        # used in most experiments
        """
        g = tf.reduce_sum(shells, axis=[2, -1], keepdims=True)
        g = tf.reduce_mean(g, axis=1, keepdims=True)
        sh_patches = tf.divide(sh_patches, tf.maximum(g, 0.000001))
        """


        # L2 norm
        l2_norm = tf.reduce_sum(tf.multiply(sh_patches, sh_patches), axis=2, keepdims=True)
        # l2_norm = tf.reduce_mean(l2_norm, axis=1, keepdims=True)
        l2_norm = tf.split(l2_norm, num_or_size_splits=self.split_size, axis=-2)
        Y = []
        for l in range(len(l2_norm)):
            ml = tf.reduce_sum(l2_norm[l], axis=-2, keepdims=True)
            ml = tf.sqrt(ml)
            Y.append(ml)
        l2_norm = tf.concat(Y, axis=-2)
        l2_norm = tf.reduce_mean(l2_norm, axis=1, keepdims=True)
        l2_norm = tf.maximum(l2_norm, 1e-8)
        l2_norm = tf.gather(l2_norm, self.sh_idx, axis=-2)
        sh_patches = tf.divide(sh_patches, l2_norm)


        """
        # L2 norm
        l2_norm = tf.reduce_sum(tf.multiply(sh_patches, sh_patches), axis=2, keepdims=True)
        l2_norm = tf.sqrt(l2_norm)
        l2_norm = tf.split(l2_norm, num_or_size_splits=self.split_size, axis=-2)
        Y = []
        for l in range(len(l2_norm)):
            ml = tf.reduce_mean(l2_norm[l], axis=-2, keepdims=True)
            Y.append(ml)
        l2_norm = tf.concat(Y, axis=-2)
        l2_norm = tf.reduce_mean(l2_norm, axis=1, keepdims=True)
        l2_norm = tf.maximum(l2_norm, 1e-8)
        l2_norm = tf.gather(l2_norm, self.sh_idx, axis=-2)
        sh_patches = tf.divide(sh_patches, l2_norm)
        """

        """
        # L1 norm
        l1_norm = tf.reduce_sum(tf.abs(sh_patches), axis=2, keepdims=True)
        l1_norm = tf.split(l1_norm, num_or_size_splits=self.split_size, axis=-2)
        Y = []
        for l in range(len(l1_norm)):
            ml = tf.reduce_mean(l1_norm[l], axis=-2, keepdims=True)
            Y.append(ml)
        l1_norm = tf.concat(Y, axis=-2)
        l1_norm = tf.reduce_mean(l1_norm, axis=1, keepdims=True)
        l1_norm = tf.maximum(l1_norm, 1e-8)
        l1_norm = tf.gather(l1_norm, self.sh_idx, axis=-2)
        sh_patches = tf.divide(sh_patches, l1_norm)
        """

        return sh_patches

class ShGaussianKernelConv(tf.keras.layers.Layer):
    def __init__(self, l_max, l_max_out=None, transpose=False, num_source_points=None):
        super(ShGaussianKernelConv, self).__init__()
        self.l_max = l_max
        self.split_size = []
        for l in range(l_max + 1):
            self.split_size.append(2 * l + 1)
        # self.output_type = output_type
        self.l_max_out = l_max_out
        self.transpose = transpose
        self.num_source_points = num_source_points
        self.Q = tf_clebsch_gordan_decomposition(l_max=max(l_max_out, l_max),
                                                 sparse=False,
                                                 output_type='dict',
                                                 l_max_out=l_max_out)

    def build(self, input_shape):
        super(ShGaussianKernelConv, self).build(input_shape)

    def call(self, x):
        assert (isinstance(x, dict))

        signal = []
        features_type = []
        channels_split_size = []
        for l in x:
            if l.isnumeric():
                features_type.append(int(l))
                # channels_split_size .append(x[l].shape[-2]*x[l].shape[-1])
                # signal.append(tf.reshape(x[l], (x[l].shape[0], -1)))
                channels_split_size.append(x[l].shape[-2] * x[l].shape[-1])
                signal.append(tf.reshape(x[l], (x[l].shape[0], x[l].shape[1], -1)))


        signal = tf.concat(signal, axis=-1)
        batch_size = signal.shape[0]
        patch_size = x["kernels"].shape[2]
        num_shells = x["kernels"].shape[-1]

        if self.transpose:
            assert(self.num_source_points is not None)
            num_points_target = self.num_source_points
            kernels = tf.reshape(x["kernels"], (batch_size, x["kernels"].shape[1], patch_size, -1, 1))
            signal = tf.reshape(signal, (signal.shape[0], signal.shape[1], 1, 1, -1))
            y = tf.multiply(signal, kernels)
            y = tf.scatter_nd(indices=x["patches idx"], updates=y,
                              shape=(batch_size, num_points_target, kernels.shape[-2], signal.shape[-1]))
        else:
            if "patches idx" in x:
                signal = tf.gather_nd(signal, x["patches idx"])

            num_points_target = signal.shape[1]
            # signal = tf.expand_dims(signal, axis=1)
            kernels = tf.reshape(x["kernels"], (batch_size, num_points_target, patch_size, -1))

            """
            signal_mean = tf.reduce_mean(signal, axis=2, keepdims=True)
            signal = tf.subtract(signal, signal_mean)
            """
            y = tf.einsum('bvpy,bvpc->bvyc', kernels, signal)



        # split y
        y_ = tf.split(y, num_or_size_splits=channels_split_size, axis=-1)
        y = {str(j): [] for j in range(self.l_max_out + 1)}
        y_cg = []
        for i in range(len(channels_split_size)):
            l = features_type[i]
            # yi = tf.reshape(y[i], (self._build_input_shape[str(l)][0], -1, self._build_input_shape[str(l)][-1]))
            yi = tf.reshape(y_[i], (batch_size, num_points_target, -1, num_shells, 2 * l + 1, x[str(l)].shape[-1]))
            yi = tf.transpose(yi, (0, 1, 2, 4, 3, 5))
            yi = tf.reshape(yi, (batch_size, num_points_target, -1, 2 * l + 1, num_shells*x[str(l)].shape[-1]))
            yi = tf.split(yi, num_or_size_splits=self.split_size, axis=2)
            for j in range(len(self.split_size)):
                # yij = tf.transpose(yi[j], (0, 2, 1, 3))
                # yij = tf.reshape(yi[j], (batch_size, num_points_target, 2 * j + 1, 2 * l + 1, -1))
                yij = yi[j]
                if l == 0:
                    y[str(j)].append(yij[:, :, :, 0, :])
                elif j == 0:
                    y[str(l)].append(yij[:, :, 0, :, :])
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

def zernike_multiplicity(d):
    m = [0]*(d+1)
    for n in range(d + 1):
        for l in range(n + 1):
            if (n - l) % 2 == 0:
                m[l] += 1
    return m

def zernike_split_size(d):
    m = zernike_multiplicity(d)
    s = []
    for l in range(len(m)):
        s.append((2*l+1)*m[l])
    return s

def spherical_harmonics_to_zernike_format(x, split=True):
    l_max = 0
    for l in x:
        if l.isnumeric():
            l_max = max(l_max, int(l))
    m = zernike_multiplicity(l_max)
    stack = not split
    if stack:
        y = []
    else:
        y = dict()
    for l in x:
        if l.isnumeric():
            sl = list(x[l].shape)
            assert (x[l].shape[-1] % m[int(l)] == 0)
            if stack:
                sl[-1] = int(sl[-1] / m[int(l)])
                sl[-2] = int(sl[-2] * m[int(l)])
                y.append(tf.reshape(x[l], sl))
            else:
                sl = sl[:-1]
                yl = tf.reshape(x[l], sl + [m[int(l)], -1])
                y[l] = yl
    if stack:
        y = tf.concat(y, axis=-2)
    return y

def split_spherical_harmonics_coeffs(x):
    l_max = int(np.sqrt(x.shape[-2]) - 0.99)
    split_size = []
    for l in range(l_max + 1):
        split_size.append(2*l+1)
    return tf.split(x, num_or_size_splits=split_size, axis=-2)

def split_zernike_coeffs(x, d):
    s = zernike_split_size(d)
    return tf.split(x, num_or_size_splits=s, axis=-2)

def spherical_harmonics_format(x):
    y = dict()
    m = zernike_multiplicity(len(x)-1)


    for l in range(len(x)):
        sl = list(x[l].shape)

        sl[-2] = int(sl[-2] / m[l])
        sl[-1] = -1

        y[str(l)] = tf.reshape(x[l], sl)
    return y

def zernike_eval(coeffs, z):
    """
    evaluate zernike functions given their coeffs
    z = Zernike(x) where x are the evaluation points
    coeffs are given in a splited spherical harmonics format
    """
    # convert coeffs to Zernike format
    z_coeffs = spherical_harmonics_to_zernike_format(coeffs, split=False)
    return tf.einsum('bpz,bvzc->bvpc', z, z_coeffs)


def spherical_harmonics_coeffs(values, z, d):
    z_coeffs = tf.einsum('bpz,bvpc->bvzc', z, values)
    z_coeffs = z_coeffs / float(z.shape[1])
    z_coeffs = split_zernike_coeffs(z_coeffs, d)
    y = spherical_harmonics_format(z_coeffs)
    return y

class ZernikePolynomials(tf.keras.layers.Layer):
    def __init__(self, d, split=True, gaussian_scale=None):
        super(ZernikePolynomials, self).__init__()
        self.d = d
        self.split = split
        self.gaussian_scale = gaussian_scale
        # self.add_channel_axis = add_channel_axis
        self.monoms_idx = tf_monomial_basis_3D_idx(d)
        Z = tf_zernike_kernel_basis(d)
        self.split_size = []
        self.Z = []
        k = 0
        for l in range(len(Z)):
            self.split_size.append(Z[l].shape[0] * Z[l].shape[1])
            self.Z.append(tf.reshape(Z[l], (-1, Z[l].shape[-1])))
        self.Z = tf.concat(self.Z, axis=0)
    def build(self, input_shape):
        super(ZernikePolynomials, self).build(input_shape)

    def call(self, x):
        monoms = tf_eval_monom_basis(x, self.d, idx=self.monoms_idx)
        z = tf.einsum('ij,...j->...i', self.Z, monoms)

        if self.gaussian_scale is not None:
            # c = tf.reduce_mean(x, axis=-2, keepdims=True)
            # x = tf.subtract(x, c)
            n2 = tf.multiply(x, x)
            n2 = tf.reduce_sum(n2, axis=-1, keepdims=True)
            g = tf.exp(-self.gaussian_scale*n2)
            z = tf.multiply(g, z)


        #  if self.add_channel_axis:
            # z = tf.expand_dims(z, axis=-1)
        if self.split:
            z = tf.split(z, num_or_size_splits=self.split_size, axis=-1)
        return z

class SphericalHarmonics(tf.keras.layers.Layer):
    def __init__(self, l_max, split=True, add_channel_axis=True):
        super(SphericalHarmonics, self).__init__()
        self.l_max = l_max
        self.split = split
        self.add_channel_axis = add_channel_axis
        self.monoms_idx = tf_monomial_basis_3D_idx(l_max)
        self.Y = tf_spherical_harmonics_basis(l_max=l_max, concat=True)
        self.split_size = []
        for l in range(l_max+1):
            self.split_size.append(2*l+1)
    def build(self, input_shape):
        super(SphericalHarmonics, self).build(input_shape)

    def call(self, x):
        x = tf.math.l2_normalize(x, axis=-1)
        monoms = tf_eval_monom_basis(x, self.l_max, idx=self.monoms_idx)
        y = tf.einsum('ij,...j->...i', self.Y, monoms)
        if self.add_channel_axis:
            y = tf.expand_dims(z, axis=-1)
        if self.split:
            y = tf.split(y, num_or_size_splits=self.split_size, axis=-1)
        return y

class SphericalHarmonicsShellsKernels(tf.keras.layers.Layer):
    def __init__(self, l_max, stack=True):
        super(SphericalHarmonicsShellsKernels, self).__init__()
        self.l_max = l_max
        self.stack = stack
        self.monoms_idx = tf_monomial_basis_3D_idx(l_max)
        self.gaussian_scale = 4*0.69314718056*(l_max**2)
        self.Y = tf_spherical_harmonics_basis(l_max, concat=True)
        self.split_size = []
        self.sh_idx = []
        for l in range(l_max + 1):
            self.split_size.append(2*l+1)


    def build(self, input_shape):
        super(SphericalHarmonicsShellsKernels, self).build(input_shape)

    def call(self, x):
        if "patches dist" in x:
            patches_dist = tf.expand_dims(x["patches dist"], axis=-1)
        else:
            patches_dist = tf.norm(x["patches"], axis=-1, keepdims=True)

        normalized_patches = tf.divide(x["patches"], tf.maximum(patches_dist, 0.000001))
        monoms_patches = tf_eval_monom_basis(normalized_patches, self.l_max, idx=self.monoms_idx)
        sh_patches = tf.einsum('ij,...j->...i', self.Y, monoms_patches)
        shells_rad = tf.range(self.l_max+1, dtype=tf.float32) / self.l_max

        shells_rad = tf.reshape(shells_rad, (1, 1, 1, -1))
        shells = tf.subtract(patches_dist, shells_rad)
        shells = tf.exp(-self.gaussian_scale*tf.multiply(shells, shells))
        shells_sum = tf.reduce_sum(shells, axis=-1, keepdims=True)
        shells = tf.divide(shells, tf.maximum(shells_sum, 0.000001))
        g = tf.reduce_sum(shells, axis=2, keepdims=True)
        g = tf.reduce_mean(g, axis=[1, -1], keepdims=True)
        shells = tf.divide(shells, tf.maximum(g, 0.000001))
        sh_patches = tf.expand_dims(sh_patches, axis=-1)
        sh_patches = tf.split(sh_patches, num_or_size_splits=self.split_size, axis=-2)
        shells = tf.expand_dims(shells, axis=-2)



        sh_shells_patches = []
        for l in range(len(sh_patches)):
            sh_shells_patches.append(tf.multiply(shells[..., l:], sh_patches[l]))

        if self.stack:
            for l in range(len(sh_shells_patches)):
                sl = list(sh_shells_patches[l].shape)
                sl = sl[:-1]
                sl[-1] = -1
                sh_shells_patches[l] = tf.reshape(sh_shells_patches[l], sl)
            sh_shells_patches = tf.concat(sh_shells_patches, axis=-1)
        return sh_shells_patches


class SphericalHarmonicsShellsConv(tf.keras.layers.Layer):
    def __init__(self, l_max, l_max_out=None):
        super(SphericalHarmonicsShellsConv, self).__init__()
        self.l_max = l_max
        self.split_size = []
        for l in range(l_max + 1):
            self.split_size.append((2 * l + 1)*(l_max + 1 - l))
        self.l_max_out = l_max_out
        self.Q = tf_clebsch_gordan_decomposition(l_max=l_max, sparse=False, output_type='dict', l_max_out=l_max_out)

    def build(self, input_shape):
        super(SphericalHarmonicsShellsConv, self).build(input_shape)

    def call(self, x):
        assert (isinstance(x, dict))

        signal = []
        features_type = []
        channels_split_size = []
        for l in x:
            if l.isnumeric():
                features_type.append(int(l))
                # channels_split_size .append(x[l].shape[-2]*x[l].shape[-1])
                # signal.append(tf.reshape(x[l], (x[l].shape[0], -1)))
                channels_split_size.append(x[l].shape[-2] * x[l].shape[-1])
                signal.append(tf.reshape(x[l], (x[l].shape[0], x[l].shape[1], -1)))

        if "patches idx" in x:
            signal = tf.concat(signal, axis=-1)
        signal = tf.gather_nd(signal, x["patches idx"])

        batch_size = signal.shape[0]
        num_points_target = signal.shape[1]

        # signal = tf.expand_dims(signal, axis=1)
        # kernels = tf.reshape(x["kernels"], (batch_size, num_points_target, patch_size, -1))
        kernels = x["kernels"]
        y = tf.einsum('bvpy,bvpc->bvyc', kernels, signal)
        # split y
        y_ = tf.split(y, num_or_size_splits=channels_split_size, axis=-1)
        y = {str(j): [] for j in range(self.l_max + 1)}
        y_cg = []
        for i in range(len(channels_split_size)):
            l = features_type[i]
            yi = tf.split(y_[i], num_or_size_splits=self.split_size, axis=2)
            for j in range(len(self.split_size)):
                yij = tf.reshape(yi[j], (batch_size, num_points_target, 2*j+1, self.l_max + 1 - j, 2*l+1, -1))
                yij = tf.transpose(yij, (0, 1, 2, 4, 3, 5))
                yij = tf.reshape(yij, (batch_size, num_points_target, 2*j+1, 2*l+1, -1))
                if l == 0:
                    y[str(j)].append(yij[:, :, :, 0, :])
                elif j == 0:
                    y[str(l)].append(yij[:, :, 0, :, :])
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


if __name__=="__main__":

    filename = "/home/rahul/research/data/sapien_processed/train_refrigerator.h5"
    f = h5py.File(filename, "r")
    x = tf.cast(f["data"][:2], dtype=tf.float32)
    x2 = tf.cast(f["data"][2:4], dtype=tf.float32)
    gi = GroupPoints(0.2, 32)
    out = gi({"source points": x, "target points": x2})
    # x = tf.ones((1, 4, 3), dtype=tf.float32) * tf.expand_dims(tf.expand_dims(tf.range(4, dtype=tf.float32), axis = 0), axis = -1)
    # y = tf_eval_monom_basis(x, 3)
    # print(x)
    # print(y, y.shape)
    sph_kernels = SphericalHarmonicsGaussianKernels(l_max = 3, gaussian_scale = 0.1, num_shells = 3)
    # x = (tf.ones((1, 100, 32, 3), dtype=tf.float32) * tf.reshape(tf.range(3, dtype=tf.float32), (1, 1, 1, -1)))
    # print(x.shape)
    patches = sph_kernels({"patches": out["patches source"], "patches dist": out["patches dist source"]})
    # print(patches, patches.shape)
    conv_layer = ShGaussianKernelConv(l_max=3, l_max_out=3)
    y = {}
    y["source points"] = x
    y["target points"] = x2
    y["patches idx"] = out["patches idx source"]
    y["patches dist source"] = out["patches dist source"]
    y["kernels"] = patches
    # w = gauss_normalization(y["patches dist source"], 1./3.)



    if '1' in y:
        y['1'] = torch.cat([y['1'], x2], dim=-1)
    else:
        y['1'] = tf.expand_dims(x2, axis=-1)

    y = conv_layer(y)
    for key in y:
        print(y[key], " ", key, " ", y[key].shape)
    