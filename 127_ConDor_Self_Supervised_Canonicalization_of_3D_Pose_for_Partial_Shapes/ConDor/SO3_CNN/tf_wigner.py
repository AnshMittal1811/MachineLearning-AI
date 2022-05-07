import numpy as np
import scipy
from sympy import *
import tensorflow as tf
# from scipy import linalg, matrix, special
# from spherical_harmonics.np_spherical_harmonics import complex_sh_, real_sh_, complex_to_real_sh
# from sympy.physics.quantum.spin import Rotation
from scipy.spatial.transform.rotation import Rotation
from SO3_CNN.sampling import SO3_sampling_from_S2, tf_polyhedrons
from circle_bundle_net.kernels import tf_complex_powers


"""
computes the coefficients of a list of polynomials in the monomial basis
"""
def np_monomial_basis_coeffs(polynomials, monoms_basis):
    n_ = len(monoms_basis)
    m_ = len(polynomials)
    M = np.zeros((m_, n_))
    for i in range(m_):
        for j in range(n_):
            M[i, j] = re(polynomials[i].coeff_monomial(monoms_basis[j]))
    return M

def tf_monomial_basis_coeffs(polynomials, monoms_basis, dtype=tf.float32):
    return tf.convert_to_tensor(np_monomial_basis_coeffs(polynomials, monoms_basis), dtype=dtype)

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def monom_deg(m):
    d = 0
    for p in m:
        d += p
    return d

def deg_sort(monoms_idx):
    idx = argsort([(monom_deg(m),)+m for m in monoms_idx])
    m = []
    for i in range(len(monoms_idx)):
        m.append(monoms_idx[idx[i]])
    return m


def monomial_basis_2D_size(d):
    num = 0
    for I in range((d + 1) ** 2):
        i = I % (d + 1)
        a = int((I - i) / (d + 1))
        j = a % (d + 1)
        if (i + j <= d):
            num += 1
    return num

def monomial_basis_2D_sizes(d_max):
    num = []
    for d in range(d_max+1):
        num.append(monomial_basis_2D_size(d))
    return num

"""
computes degree d monoms in x, y
"""
def monoms_2D(d):
    monoms_basis = []
    for I in range((d + 1) ** 2):
        i = I % (d + 1)
        a = int((I - i) / (d + 1))
        j = a % (d + 1)
        if (i + j == d):
            monoms_basis.append((i, j))
    monoms_basis = list(set(monoms_basis))
    monoms_basis = sorted(monoms_basis)
    return monoms_basis

"""
evaluate degree d monoms in x, y on an input with axis of size 2
"""
def tf_eval_monoms(x, d, axis=-1):
    m = monoms_2D(d)
    pows = np.zeros((2, len(m)), dtype=np.int32)
    for i in range(len(m)):
        for j in range(2):
            pows[j, i] = m[i][j]
    pows = tf.convert_to_tensor(pows, dtype=tf.float32)
    n = len(list(x.shape))
    axis = axis % n
    shape = [1]*(n+1)
    shape[axis] = 2
    shape[-1] = len(m)
    pows = tf.reshape(pows, shape)
    x = tf.expand_dims(x, axis=-1)
    y = tf.pow(x, pows)
    y = tf.reduce_prod(y, axis, keepdims=False)

    return y





"""
computes the monomial basis in x, y up to degree d
"""
def monomial_basis_2D(d, deg_sort_=True):
    monoms_basis = []
    for I in range((d + 1) ** 2):
        i = I % (d + 1)
        a = int((I - i) / (d + 1))
        j = a % (d + 1)
        if (i + j <= d):
            monoms_basis.append((i, j))

    monoms_basis = list(set(monoms_basis))
    monoms_basis = sorted(monoms_basis)
    if deg_sort_:
        monoms_basis = deg_sort(monoms_basis)
    return monoms_basis

def tf_monomial_basis_2D_idx(d):
    m = monomial_basis_2D(d)
    idx = np.zeros((len(m), 2), dtype=np.int32)
    for i in range(len(m)):
        for j in range(2):
            idx[i, j] = m[i][j]
    return tf.convert_to_tensor(idx, dtype=tf.int64)

"""
evaluate monomial basis up to degree d
"""
def tf_eval_monom_basis(x, d, idx=None):

    if idx is None:
        idx = tf_monomial_basis_2D_idx(d)
    y = []
    for i in range(2):
        pows = tf.reshape(tf.range(d+1, dtype=tf.float32), [1]*len(x.shape) + [d+1])
        xi = tf.expand_dims(x[..., i], axis=-1)
        yi = tf.pow(xi, pows)
        y.append(tf.gather(yi, idx[..., i], axis=-1))
    y = tf.stack(y, axis=-1)
    y = tf.reduce_prod(y, axis=-1, keepdims=False)
    return y

def fourier_basis(l, x, y):
    c = []
    s = []
    for k in range(l+1):
        f = (x + I*y)**k
        f = expand(f)
        c.append(Poly(re(f), x, y))
        s.append(Poly(im(f), x, y))
    return c, s

def tf_fourier_basis_(l, stack=True):
    m = monomial_basis_2D(l)
    x, y = symbols("x y", real=True)
    c, s = fourier_basis(l, x, y)
    c = tf_monomial_basis_coeffs(c, m)
    s = tf_monomial_basis_coeffs(s, m)
    if stack:
        return tf.stack([c, s], axis=-2)
    else:
        return c, s

def tf_fourier_basis(l_max, c, s, stack=True):
    e = tf.dtypes.complex(c, s)
    e = tf.expand_dims(e, axis=-1)
    shape = [1]*len(list(e.shape))
    shape[-1] = l_max+1
    pows = tf.reshape(tf.cast(tf.range(l_max+1, dtype=tf.int32), tf.complex64), shape)
    e = tf.pow(e, pows)
    c = tf.math.real(e)
    s = tf.math.imag(e)
    if stack:
        return tf.stack([c, s], axis=-1)
    else:
        return c, s


def jacobi_polynomial(alpha, beta, n, z):
    K = gamma(alpha + n + 1) / (factorial(n)*gamma(alpha+beta+n+1))
    P = 0
    for m in range(n+1):
        P += binomial(n, m)*(gamma(alpha+beta+n+m+1)/gamma(alpha+m+1))((z-1) / 2.)**m
    return K*P


def wigner_d_matrix_coeffs(l, j, k, b):
    p = np.math.factorial(l+j)*np.math.factorial(l-j)*np.math.factorial(l+k)*np.math.factorial(l-k)
    p = np.sqrt(p)

    # l + k - s >= 0
    # s >= 0
    # j - k + s >= 0
    # l - j - s >= 0

    # l + k >= s
    # s >= 0
    # s >= k - j
    # l - j >= s

    s1 = np.max([0, k-j])
    s2 = np.min([l+k, l-j])
    s_ = np.sin(b/2.)
    c_ = np.cos(b/2.)
    d = 0.
    for s in range(s1, s2+1):
        q = np.math.factorial(l+k-s)*np.math.factorial(s)*np.math.factorial(j-k+s)*np.math.factorial(l-j-s)
        x = (1.*p)/(1.*q)
        x *= (-1)**(j-k+s)
        x *= (c_**(2*l+k-j-2*s))*(s_**(j-k+2*s))
        d += x
    return d



def wigner_d_matrix_polynomial(l, j, k, x, y):
    p = np.math.factorial(l + j) * np.math.factorial(l - j) * np.math.factorial(l + k) * np.math.factorial(l - k)
    p = np.sqrt(p)

    # l + k - s >= 0
    # s >= 0
    # j - k + s >= 0
    # l - j - s >= 0

    # l + k >= s
    # s >= 0
    # s >= k - j
    # l - j >= s

    s1 = np.max([0, k - j])
    s2 = np.min([l + k, l - j])
    # s = np.sin(b / 2.)
    # c = np.cos(b / 2.)
    d = 0.
    for s in range(s1, s2 + 1):
        q = np.math.factorial(l + k - s) * np.math.factorial(s) * np.math.factorial(j - k + s) * np.math.factorial(
            l - j - s)
        a = (1. * p) / (1. * q)
        a *= (-1.) ** (j - k + s)
        a *= (x ** (2 * l + k - j - 2 * s)) * (y ** (j - k + 2 * s))
        d += a
    # print(d)
    # d = Poly(expand(d), x, y)
    return d

def real_to_complex_sh(l):
    C = np.zeros(shape=(2*l+1, 2*l+1), dtype=np.complex64)
    c = 1./np.sqrt(2.)
    for m in range(1, l+1):
        C[l + m, l + m] = -1j * c
        C[l + m, l - m] = c
    for m in range(-l, 0):
        C[l + m, l + m] = ((-1)**m)*c
        C[l + m, l - m] = 1j*((-1) ** m)*c

    C[l, l] = 1.
    C = np.flip(C, 0)
    C = np.flip(C, 1)
    # print(C)
    return np.asmatrix(C)


def tf_wigner_d_matrix_(l, matrix_format=True):
    m = monoms_2D(2*l)
    d = []
    x, y = symbols("x y")

    d = Matrix(np.zeros((2 * l + 1, 2 * l + 1)))

    for j in range(2*l+1):
        for k in range(2*l+1):
            # d.append(wigner_d_matrix_polynomial(l, j-l, k-l, x, y))
            d[j, k] = wigner_d_matrix_polynomial(l, j-l, k-l, x, y)

    C = real_to_complex_sh(l)

    R = np.conjugate(C.T)


    C = Matrix(C)
    R = Matrix(R)
    d = R*(d*C)

    L = []
    for j in range(2*l+1):
        for k in range(2*l+1):
            L.append(Poly(simplify(expand(d[j, k])), x, y))
    print(L)
    # d[0] = Poly(x**2 + y**2, x, y)
    coeffs = tf_monomial_basis_coeffs(L, m)
    if matrix_format:
        coeffs = tf.reshape(coeffs, (2*l+1, 2*l+1, -1))
    return coeffs

def matrix_to_euler_zyz_angles(x):

    # alpha = tf.atan2(x[..., 1, 2], x[..., 0, 2])
    # gamma = tf.atan2(x[..., 2, 1], -x[..., 2, 0])

    alpha = tf.stack([x[..., 0, 2], x[..., 1, 2]], axis=-1)
    alpha = tf.linalg.l2_normalize(alpha, axis=-1)

    gamma = tf.stack([-x[..., 2, 0], x[..., 2, 1]], axis=-1)
    gamma = tf.linalg.l2_normalize(gamma, axis=-1)

    beta = tf.atan2(tf.multiply(alpha[..., 0], x[..., 0, 2]) + tf.multiply(alpha[..., 1], x[..., 1, 2]), x[..., 2, 2])
    beta = tf.stack([tf.cos(beta / 2), tf.sin(beta / 2)], axis=-1)

    rot_z = tf.stack([x[..., 0, 0], x[..., 1, 0]], axis=-1)
    shape = list(x.shape)
    shape = shape[:-2]
    zero = tf.stack([tf.ones(shape), tf.zeros(shape)], axis=-1)
    cond = tf.expand_dims(tf.greater(tf.abs(x[..., 2, 2]), 0.99), axis=-1)

    alpha = tf.where(cond, rot_z, alpha)
    beta = tf.where(cond, zero, beta)
    gamma = tf.where(cond, zero, gamma)
    return alpha, beta, gamma

def z_wigner(c, s):
    c = tf.linalg.diag(c)
    s = tf.reverse(tf.linalg.diag(s), axis=[-1])
    return c+s
def compute_wigner_matrix(d, alpha, beta, gamma):
    ca = tf.concat([tf.reverse(alpha[..., 1:, 0], axis=[-1]), alpha[..., 0]], axis=-1)
    ca = tf.expand_dims(ca, axis=-1)
    sa = tf.concat([-tf.reverse(alpha[..., 1:, 1], axis=[-1]), alpha[..., 1]], axis=-1)
    sa = tf.expand_dims(sa, axis=-1)

    cc = tf.concat([tf.reverse(gamma[..., 1:, 0], axis=[-1]), gamma[..., 0]], axis=-1)
    cc = tf.expand_dims(cc, axis=-2)
    sc = tf.concat([-tf.reverse(gamma[..., 1:, 1], axis=[-1]), gamma[..., 1]], axis=-1)
    sc = tf.expand_dims(sc, axis=-2)


    d = tf.einsum('ijk,...k->...ij', d, beta)
    d = tf.multiply(ca, d) + tf.multiply(-sa, tf.reverse(d, axis=[-2]))
    d = tf.multiply(cc, d) + tf.multiply(sc, tf.reverse(d, axis=[-1]))


    return d




def rot_z(t):
    c = tf.cos(t)
    s = tf.sin(t)
    z = tf.zeros(t.shape)
    o = tf.ones(t.shape)
    r = tf.stack([c, -s, z, s, c, z, z, z, o], axis=-1)
    r = tf.reshape(r, list(t.shape) + [3, 3])
    return r

def rot_y(t):
    c = tf.cos(t)
    s = tf.sin(t)
    z = tf.zeros(t.shape)
    o = tf.ones(t.shape)
    r = tf.stack([c, z, s, z, o, z, -s, z, c], axis=-1)
    r = tf.reshape(r, list(t.shape) + [3, 3])
    return r

def zyz_euler_angles_rot(a, b, c):
    ra = rot_z(a)
    rb = rot_y(b)
    rc = rot_z(c)
    r = tf.matmul(tf.matmul(ra, rb), rc)
    return r

def zyz_euler_angles(a, b, c):
    ca = tf.cos(a)
    sa = tf.sin(a)
    cb = tf.cos(b/2.)
    sb = tf.sin(b/2.)
    cc = tf.cos(c)
    sc = tf.sin(c)
    a = tf.stack([ca, sa], axis=-1)
    b = tf.stack([cb, sb], axis=-1)
    c = tf.stack([cc, sc], axis=-1)
    zyz = tf.stack([a, b, c], axis=-1)
    return zyz


class tf_wigner_matrix:
    def __init__(self, l_max=3, l_list=None):
        if l_list is None:
            self.l_list = range(l_max+1)
        else:
            self.l_list = l_list

        self.l_max = max(self.l_list)
        self.monom_basis_size = monomial_basis_2D_sizes(2*l_max)
        self.d = dict()
        #
        self.f = tf_fourier_basis_(l_max)
        for l in self.l_list:
            self.d[str(l)] = tf_wigner_d_matrix_(l, matrix_format=True)


    def compute(self, x):
        def prepare_fourier(t, f):
            # t = tf.stack([tf.cos(t), tf.sin(t)], axis=-1)
            t = tf_eval_monom_basis(t, self.l_max, idx=None)
            t = tf.einsum('ijk,...k->...ij', f, t)
            ct = tf.concat([tf.reverse(t[..., 1:, 0], axis=[-1]), t[..., 0]], axis=-1)
            st = tf.concat([-tf.reverse(t[..., 1:, 1], axis=[-1]), t[..., 1]], axis=-1)
            return tf.stack([ct, st], axis=-1)

        alpha, beta, gamma = matrix_to_euler_zyz_angles(x)

        # print(tf.atan2(alpha[..., 1], alpha[..., 0]))
        # print(2*tf.atan2(beta[..., 1], beta[..., 0]))
        # print(tf.atan2(gamma[..., 1], gamma[..., 0]))



        # alpha = prepare_fourier(alpha, self.f)
        alpha = tf_fourier_basis(self.l_max, alpha[..., 0], alpha[..., 1])



        # gamma = prepare_fourier(gamma, self.f)
        gamma = tf_fourier_basis(self.l_max, gamma[..., 0], gamma[..., 1])
        # L = 2*self.l_max + 1
        y = dict()
        for l in self.l_list:
            b = tf_eval_monoms(beta, 2 * l)

            if l == 0:
                a = tf.expand_dims(alpha[..., 0, :], axis=-2)
                c = tf.expand_dims(gamma[..., 0, :], axis=-2)
            else:
                a = alpha[..., :l+1, :]
                c = gamma[..., :l+1, :]

            y[str(l)] = compute_wigner_matrix(self.d[str(l)], a, b, c)
        return y

    def compute_euler_zyz(self, zyz):

        zyz = zyz_euler_angles(zyz[..., 0], zyz[..., 1], zyz[..., 2])

        alpha = zyz[..., 0]
        beta = zyz[..., 1]
        gamma = zyz[..., 2]

        # print(tf.atan2(alpha[..., 1], alpha[..., 0]))
        # print(2*tf.atan2(beta[..., 1], beta[..., 0]))
        # print(tf.atan2(gamma[..., 1], gamma[..., 0]))



        # alpha = prepare_fourier(alpha, self.f)
        alpha = tf_fourier_basis(self.l_max, alpha[..., 0], alpha[..., 1])



        # gamma = prepare_fourier(gamma, self.f)
        gamma = tf_fourier_basis(self.l_max, gamma[..., 0], gamma[..., 1])
        y = dict()
        for l in self.l_list:
            b = tf_eval_monoms(beta, 2 * l)

            if l == 0:
                a = tf.expand_dims(alpha[..., 0, :], axis=-2)
                c = tf.expand_dims(gamma[..., 0, :], axis=-2)
            else:
                a = alpha[..., :l+1, :]
                c = gamma[..., :l+1, :]

            y[str(l)] = compute_wigner_matrix(self.d[str(l)], a, b, c)
        return y




class tf_complex_wigner_matrix:
    def __init__(self, l_max=3, l_list=None):
        if l_list is None:
            self.l_list = range(l_max+1)
        else:
            self.l_list = l_list

        self.l_max = max(self.l_list)
        self.monom_basis_size = monomial_basis_2D_sizes(2*l_max)
        self.d = dict()
        #
        self.f = tf_fourier_basis_(l_max)
        for l in self.l_list:
            self.d[str(l)] = tf_wigner_d_matrix_(l, matrix_format=True)


    def compute(self, x):
        alpha, beta, gamma = matrix_to_euler_zyz_angles(x)

        ea = tf_complex_powers(alpha[..., 0], alpha[..., 1], self.l_max)
        ec = tf_complex_powers(gamma[..., 0], gamma[..., 1], self.l_max)

        y = dict()
        for l in self.l_list:
            b = tf_eval_monoms(beta, 2 * l)
            d = tf.einsum('ijk,...k->...ij', self.d[l], b)
            d = tf.cast(d, dtype=tf.complex64)
            ea_ = ea[..., self.l_max - l:self.l_max + l]
            ec_ = ec[..., self.l_max - l:self.l_max + l]
            d = tf.multiply(tf.expand_dims(ea_, axis=-1), d)
            d = tf.multiply(tf.expand_dims(ec_, axis=-2), d)
            y[str(l)] = d
        return y

    def compute_euler_zyz(self, zyz):

        zyz = zyz_euler_angles(zyz[..., 0], zyz[..., 1], zyz[..., 2])

        alpha = zyz[..., 0]
        beta = zyz[..., 1]
        gamma = zyz[..., 2]

        ea = tf_complex_powers(alpha[..., 0], alpha[..., 1], self.l_max)
        ec = tf_complex_powers(gamma[..., 0], gamma[..., 1], self.l_max)

        y = dict()
        for l in self.l_list:
            b = tf_eval_monoms(beta, 2 * l)
            d = tf.einsum('ijk,...k->...ij', self.d[l], b)
            d = tf.cast(d, dtype=tf.complex64)
            ea_ = ea[..., self.l_max - l:self.l_max + l]
            ec_ = ec[..., self.l_max - l:self.l_max + l]
            d = tf.multiply(tf.expand_dims(ea_, axis=-1), d)
            d = tf.multiply(tf.expand_dims(ec_, axis=-2), d)
            y[str(l)] = d
        return y


def convert_to_real_(x):
    l = int(x.shape[-2] / 2.)
    if l == 0:
        return x
    c = tf.math.cumprod(tf.fill([l], -1.), axis=0)
    shape = [1]*len(list(x.shape))
    shape[-2] = l
    c = tf.reshape(c, shape)
    x0 = tf.expand_dims(x, )
    x_pos = x[..., l-1:, :]
    _x_pos = tf.multiply(c, x_pos)
    x_neg = x[..., :l, :]
    y_pos = (1.j / np.sqrt(2))*(x_neg - _x_pos)
    y_neg = (1. / np.sqrt(2))*(x_neg + _x_pos)
    y = tf.concat([y_neg, x0, y_pos], axis=-2)
    return y

def convert_to_real(x):
    y = dict()
    for l in x:
        if l.isnumeric():
            y[l] = convert_to_real_(x[l])
    return y





class WignerEval:
    def __init__(self, base='pentakis', k=8, l_max=3, l_list=None, wigner_fn=None):
        self.base = base
        self.k = k
        if wigner_fn is not None:
            if l_list is not None:
                self.l_list = l_list
                self.l_max = max(l_list)
            else:
                self.l_list = range(l_max+1)
                self.l_max = l_max
            self.wigner_fn = wigner_fn
        else:
            self.wigner_fn = tf_wigner_matrix(l_max=l_max, l_list=l_list)

        if isinstance(self.base, str):
            S2 = tf_polyhedrons(self.base)
            SO3 = SO3_sampling_from_S2(S2, self.k)
        else:
            SO3 = self.base



        d = self.wigner_fn.compute(SO3)
        self.types = d.keys()
        D = []
        for l in self.types:
            # D.append((2.*int(l)+1.)*tf.reshape(d[l], (-1, (2*int(l)+1)**2)))
            D.append(tf.reshape(d[l], (-1, (2 * int(l) + 1) ** 2)))
        self.D = tf.concat(D, axis=-1)


    def compute(self, x):
        X = []
        for l in self.types:
            sl = list(x[l].shape)
            sl[-2] = sl[-1]
            sl[-3] = -1
            sl = sl[:-1]
            X.append(tf.reshape(x[l], sl))
        X = tf.concat(X, axis=-2)
        return tf.einsum('vi,...ic->...vc', self.D, X)

class WignerCoeffs:
    def __init__(self, base='pentakis', k=8, l_max=3, l_list=None, wigner_fn=None):
        self.base = base
        self.k = k
        if l_list is not None:
            self.l_list = l_list
            self.l_max = max(l_list)
        else:
            self.l_list = range(l_max + 1)
            self.l_max = l_max
        if wigner_fn is not None:
            self.wigner_fn = wigner_fn
        else:
            self.wigner_fn = tf_wigner_matrix(l_max=self.l_max, l_list=self.l_list)

        if isinstance(self.base, str):
            S2 = tf_polyhedrons(self.base)
            SO3 = SO3_sampling_from_S2(S2, self.k)
        else:
            SO3 = self.base

        d = self.wigner_fn.compute(SO3)
        self.types = list(d.keys())
        D = []
        for l in self.types:
            D.append((2.*int(l)+1.)*tf.reshape(d[l], (-1, (2*int(l)+1)**2)))
        self.D = tf.concat(D, axis=-1)

        self.split_size = []
        for i in range(len(self.l_list)):
            self.split_size.append((2*self.l_list[i] + 1)**2)


    def compute(self, x):

        # c = tf.einsum('vi,...vc->...ic', self.D, x) / (self.D.shape[0] / (8*np.pi**2))
        c = tf.einsum('vi,...vc->...ic', self.D, x) / (self.D.shape[0])
        c = tf.split(c, num_or_size_splits=self.split_size, axis=-2)
        C = dict()
        for i in range(len(self.types)):
            l = self.types[i]
            sl = list(c[i].shape)
            nc = sl[-1]
            sl[-2] = 2*int(l)+1
            sl[-1] = 2*int(l)+1
            sl += [nc]
            C[l] = tf.reshape(c[i], sl)
        return C


def S2_to_SO3_pullback(x):
    y = dict()
    for l in x:
        if l.isnumeric():
            yl = tf.expand_dims(x[l], axis=-2)
            tiles = [1]*(len(x[l].shape)+1)
            tiles[-2] = x[l].shape[-2]
            yl = tf.tile(yl, tiles)
            y[l] = yl
    return y

def norms(x, axis=-2):
    y = []
    for l in x:
        if l.isnumeric():
            if int(l) > 0:
                yl = tf.multiply(x[l], x[l])
                yl = tf.reduce_sum(yl, axis=axis, keepdims=False)
                yl = tf.sqrt(tf.maximum(yl, 0.000001))
            else:
                yl = x[l][..., 0, :]
            s = list(yl.shape)
            s[-2] = -1
            yl = tf.reshape(yl, s[:-1])
            y.append(yl)
    y = tf.concat(y, axis=-1)
    return y

