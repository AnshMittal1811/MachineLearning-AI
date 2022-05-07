import numpy as np
import tensorflow as tf
from sympy import *
from SO3_CNN.sampling import SO3_sampling_from_S2, tf_polyhedrons
def tf_fibonnacci_sphere_sampling(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    S2 = np.stack([x, y, z], axis=-1)
    return tf.convert_to_tensor(S2, dtype=tf.float32)

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
computes (x-iy)^{l}, ... , 1 , (x +iy)^{l}
"""
def tf_complex_powers(x, y, l):
    z = tf.complex(x, y)
    n = len(list(z.shape))

    """
    m = tf.range(l + 1)
    if n > 0:
        m = tf.reshape(m, [1]*n + [l+1])
    z = tf.pow(z, m)
    """

    o = tf.ones(list(z.shape) + [1], dtype=tf.complex64)
    z = tf.expand_dims(z, axis=-1)
    z = tf.tile(z, [1]*n + [l])
    z = tf.math.cumprod(z, axis=-1)
    z = tf.concat([o, z], axis=-1)
    if l > 0:
        z_ = tf.reverse(z[..., 1:], axis=tf.convert_to_tensor(np.array([-1])))
        z_ = tf.math.conj(z_)
        z = tf.concat([z_, z], axis=-1)
    return z

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
    Ylm = Poly(simplify(expand(Ylm)), x, y, z)
    return Ylm

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
def tf_eval_monoms_2D(x, d, axis=-1):
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
computes degree d monoms in x, y, z
"""
def monoms_3D(d):
    monoms_basis = []
    for I in range((d + 1) ** 3):
        i = I % (d + 1)
        a = int((I - i) / (d + 1))
        j = a % (d + 1)
        k = int((a - j) / (d + 1))
        if (i + j + k == d):
            monoms_basis.append((i, j, k))
    monoms_basis = list(set(monoms_basis))
    monoms_basis = sorted(monoms_basis)
    return monoms_basis


"""
evaluate degree d monoms in x, y, z on an input with axis of size 2
"""
def tf_eval_monoms_3D(x, d, axis=-1):
    m = monoms_3D(d)
    pows = np.zeros((3, len(m)), dtype=np.int32)
    for i in range(len(m)):
        for j in range(3):
            pows[j, i] = m[i][j]
    pows = tf.convert_to_tensor(pows, dtype=tf.float32)
    n = len(list(x.shape))
    axis = axis % n
    shape = [1]*(n+1)
    shape[axis] = 3
    shape[-1] = len(m)
    pows = tf.reshape(pows, shape)
    x = tf.expand_dims(x, axis=-1)
    y = tf.pow(x, pows)
    y = tf.reduce_prod(y, axis, keepdims=False)
    return y

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

def tf_spherical_harmonics_(l, matrix_format=True):
    monoms = monoms_3D(l)
    sph_polys = []
    x, y, z = symbols("x y z")
    for m in range(2*l+1):
        sph_polys.append(real_spherical_harmonic(l, m-l, x, y, z))
    coeffs = tf_monomial_basis_coeffs(sph_polys, monoms)
    if matrix_format:
        coeffs = tf.reshape(coeffs, (2*l+1, -1))
    return coeffs

class tf_spherical_harmonics:
    def __init__(self, l_max=3, l_list=None):
        if l_list is None:
            self.l_list = range(l_max+1)
        else:
            self.l_list = l_list
        self.l_max = max(self.l_list)
        self.Y = dict()
        for l in self.l_list:
            self.Y[str(l)] = tf_spherical_harmonics_(l)

    def compute(self, x):
        Y = dict()
        for l in self.l_list:
            ml = tf_eval_monoms_3D(x, l)
            Y[str(l)] = tf.einsum('mk,...k->...m', self.Y[str(l)], ml)
        return Y

def tf_legendre_polynomials_(l, matrix_format=True):
    monoms = monoms_2D(l)
    legendre_polys = []
    z, r2 = symbols("x r2")
    for m in range(2*l+1):
        p = associated_legendre_polynomial(l, m-l, z, r2)
        p = Poly(simplify(expand(p)), z, r2)
        legendre_polys.append(p)
    coeffs = tf_monomial_basis_coeffs(legendre_polys, monoms)
    if matrix_format:
        coeffs = tf.reshape(coeffs, (2*l+1, -1))
    return coeffs

class tf_legendre_polynomials:
    def __init__(self, l_max=3, l_list=None):
        if l_list is None:
            self.l_list = range(l_max+1)
        else:
            self.l_list = l_list
        self.l_max = max(self.l_list)
        self.P = dict()
        for l in self.l_list:
            self.P[str(l)] = tf_legendre_polynomials_(l, matrix_format=True)

    def compute(self, z, r2):
        P = dict()
        z_r2 = tf.stack([z, r2], axis=-1)
        for l in self.l_list:
            ml = tf_eval_monoms_2D(z_r2, l)
            P[str(l)] = tf.einsum('mk,...k->...m', self.P[str(l)], ml)
        return P



class tf_complex_spherical_harmonics:
    def __init__(self, l_max=3, l_list=None, normalize=False):
        if l_list is None:
            self.l_list = range(l_max+1)
        else:
            self.l_list = l_list
        self.l_max = max(self.l_list)
        self.P = dict()
        self.P = tf_legendre_polynomials(l_max=3, l_list=None)
        self.normalize = normalize
    def compute(self, x):
        r2 = tf.reduce_sum(tf.multiply(x, x), axis=-1, keepdims=False)
        u = tf.stack([x[..., 0], x[..., 1]], axis=-1)
        u = tf.math.l2_normalize(u, axis=-1)
        if self.normalize:
            r = tf.sqrt(tf.maximum(r2, 0.000001))
            x = tf.divide(x, tf.expand_dims(r, axis=-1))
        x_iy = tf_complex_powers(u[..., 0], u[..., 1], self.l_max)
        z = x[..., -1]
        P = self.P.compute(z, r2)
        Y = dict()
        for l in self.l_list:
            Pl = tf.cast(P[str(l)], dtype=tf.complex64)
            Y[str(l)] = tf.multiply(Pl, x_iy[..., self.l_max-l:self.l_max+l])
        return Y

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
            Y.append(tf.reshape(y[l], (-1, 2*int(l)+1))) # v, sum 2 * l + 1 = 16
        self.Y = tf.concat(Y, axis=-1)

    def compute(self, x):
        X = []
        for l in self.types:
            X.append(x[l])
        X = tf.concat(X, axis=-2)
        return tf.einsum('vm,...mc->...vc', self.Y, X)


class SphericalHarmonicsCoeffs:
    def __init__(self, base='pentakis', l_max=3, l_list=None, sph_fn=None):
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

        if isinstance(self.base, str):
            S2 = tf_polyhedrons(self.base)
        else:
            S2 = self.base



        y = self.sph_fn.compute(S2)

        self.types = list(y.keys())
        Y = []
        for l in self.types:
            Y.append(tf.reshape(y[l], (-1, 2*int(l)+1)))
        self.Y = tf.concat(Y, axis=-1)
        self.S2 = S2

    def compute(self, x):
        X = []
        c = tf.einsum('vm,...vc->...mc', self.Y, x) / (self.Y.shape[0] / (4*np.pi))
        c = tf.split(c, num_or_size_splits=self.split_size, axis=-2)

        C = dict()
        for i in range(len(self.types)):
            l = self.types[i]
            sl = list(x.shape)
            sl[-2] = 2*int(l)+1
            C[l] = tf.reshape(c[i], sl)
        return C

    def get_samples(self):
        return self.S2


if __name__ == "__main__":

    S2 = tf_fibonnacci_sphere_sampling(64)
    y = {}

    for i in range(4):
        y[str(i)] = (tf.ones((2, 16, 2*i + 1, 128)) * tf.reshape(tf.range(16, dtype = tf.float32), (1, -1, 1, 1)))
    x = y.copy()
    y = SphericalHarmonicsEval(l_max=3, base=S2).compute(y)
    y = SphericalHarmonicsCoeffs(l_max=3, base=S2).compute(y)
    for key in y:
        print(y[key], y[key].shape)
    # print(y, y.shape)
