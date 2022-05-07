import numpy as np
import scipy
from sympy import *
# import tensorflow as tf
from scipy import linalg, matrix, special
# from spherical_harmonics.np_spherical_harmonics import complex_sh_, real_sh_, complex_to_real_sh
# from sympy.physics.quantum.spin import Rotation
from scipy.spatial.transform.rotation import Rotation




"""
Given a rotation matrix R and Y_{lk} the spherical harmonics basis 
for l in NN and k in [|-l, l|] we have 
Y_l( R^{-1} x) = D^l(R)Y_l(x) where D^l is the wigner matrix 
See https://en.wikipedia.org/wiki/Wigner_D-matrix

In particular complex and real version of the Wigner D matrix and 
its relation to the Wigner d matrix
"""

# change of basis from real to complex spherical harmonics basis
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

# change of basis from complex to real spherical harmonics basis
def complex_to_real_sh(l):
    return (real_to_complex_sh(l).conjugate()).T

# rotation matrix around z axis
def z_rot(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.matrix([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

# rotation matrix around y axis
def y_rot(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.matrix([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

# rotation matrix for Euler angles in z-y-z convention
def euler_rot_zyz(a, b ,c):
    return np.matmul(np.matmul(z_rot(a), y_rot(b)), z_rot(c))


def complex_wigner_2_(a, b, c):
    ea = np.exp(1j*a)
    eb = np.exp(1j*b)
    ec = np.exp(1j*c)

    e_a = np.exp(-1j*a)
    e_b = np.exp(-1j*b)
    e_c = np.exp(-1j*c)

    e2a = np.exp(1j*2.*a)
    e2b = np.exp(1j*2.*b)
    e2c = np.exp(1j*2.*c)

    e_2a = np.exp(-1j*2.*a)
    e_2b = np.exp(-1j*2.*b)
    e_2c = np.exp(-1j*2.*c)

    sa = np.imag(ea)
    ca = np.real(ea)

    # sb = np.imag(eb)
    # cb = np.real(eb)
    sb = np.sin(b)
    cb = np.cos(b)

    sc = np.imag(ec)
    cc = np.real(ec)

    # c2b = np.real(e2b)
    # s2b = np.imag(e2b)
    c2b = np.cos(2.*b)
    s2b = np.sin(2.*b)
    
    d22 = ((1+cb)*(1.+cb))/4.
    d21 = -sb*(1.+cb)/2.
    d20 = np.sqrt(3./8.)*sb*sb
    d2_1 = -sb*(1.-cb)/2.
    d2_2 = (1.-cb)*(1.-cb)/4.
    d11 = (2.*cb*cb+cb-1.)/2.
    d10 = -np.sqrt(3./8.)*s2b
    d1_1 = (-2.*cb*cb+cb+1.)/2.
    d00 = (3.*cb*cb-1.)/2.

    d = np.asmatrix([[d22, -d21, d20, -d2_1, d2_2],
                     [d21, d11, -d10, d1_1, -d2_1],
                     [d20, d10, d00, -d10, d20],
                     [d2_1, d1_1, d10, d11, -d21],
                     [d2_2, d2_1, d20, d21, d22]], dtype=np.complex64)

    # debug d

    d = d.T

    """
    for i in range(-2, 3):
        for j in range(-2, 3):
            print( str(i) + ' ' + str(j))
            print(str(np.real(d[i+2, j+2])) + ' ' + str(((-1.)**(j-i))*np.real(d[j+2, i+2])) + ' ' + str(np.real(d[2-j, 2-i])))
    """


    Ea = np.asmatrix([[e_2a, 0., 0., 0., 0.],
                      [0., e_a, 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., ea, 0.],
                      [0., 0., 0., 0., e2a]], dtype=np.complex64)

    Ec = np.asmatrix([[e_2c, 0., 0., 0., 0.],
                      [0., e_c, 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., ec, 0.],
                      [0., 0., 0., 0., e2c]], dtype=np.complex64)


    """
    Ea = np.asmatrix([[e2a, 0., 0., 0., 0.],
                      [0., ea, 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., e_a, 0.],
                      [0., 0., 0., 0., e_2a]], dtype=np.complex64)

    Ec = np.asmatrix([[e2c, 0., 0., 0., 0.],
                      [0., ec, 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., e_c, 0.],
                      [0., 0., 0., 0., e_2c]], dtype=np.complex64)
    """

    return np.matmul(np.matmul(Ea, d), Ec)

def complex_wigner_1_(a, b, c):
    cb = np.cos(b)
    sb = np.sin(b)

    ea = np.exp(1j * a)
    ec = np.exp(1j * c)

    e_a = np.exp(-1j * a)
    e_c = np.exp(-1j * c)

    d11 = (1.+cb)/2.
    d10 = -sb/(np.sqrt(2.))
    d1_1 = (1.-cb)/2.
    d00 = cb

    d = np.asmatrix([[d11, -d10, d1_1],
                     [d10, d00, -d10],
                     [d1_1, d10, d11]], dtype=np.complex64)

    d = d.T

    Ea = np.asmatrix([[e_a, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., ea]], dtype=np.complex64)

    Ec = np.asmatrix([[e_c, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., ec]], dtype=np.complex64)

    return np.matmul(np.matmul(Ea, d), Ec)





def complex_wigner_(l, a, b, c):
    assert (l == 0 or l == 1 or l == 2)
    if l == 0:
        return np.asmatrix([[1.]], dtype=np.complex64)
    if l == 1:
        return complex_wigner_1_(a, b, c)
    if l == 2:
        return complex_wigner_2_(a, b, c)

"""
compute the coefficient d^l_{jk} of the wigner d matrix encoding the action of a rotation of angle b 
around the y axis on the real (and complex) spherical harmonic basis of degree l
"""
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

"""
compute the wigner d matrix d^l encoding the action of a rotation of angle b 
around the y axis on the real (and complex) spherical harmonic basis of degree l
"""
def wigner_d_matrix(l, b, dtype=np.float32):
    d = np.zeros(shape=(2*l+1, 2*l+1), dtype=dtype)
    """
    for m in range((2*l+1)*(2*l+1)):
        k = m % (2*l+1)
        j = np.int((m - k) / (2*l+1))
        d[j, k] = wigner_d_matrix_coeffs(l, j-l, k-l, b)
    """
    for j in range(2*l+1):
        for k in range(2*l+1):
            d[j, k] = wigner_d_matrix_coeffs(l, j-l, k-l, b)
    return np.asmatrix(d)

"""
compute the action of rotation of angle a around the z axis on the 
complex spherical harmonic basis of degree l
"""
def diag_exp(l, a):
    e = np.zeros(shape=(2*l+1, 2*l+1), dtype=np.complex64)

    for m in range(l+1):
        e[m + l, m + l] = np.exp(m * 1j * a)
        e[m, m] = np.exp((m - l) * 1j * a)


    return np.asmatrix(e)


"""
def complex_D_wigner(l, a, b, c):
    D = diag_exp(l, a)*wigner_d_matrix(l, b, dtype=np.complex64)*diag_exp(l, c)
    return np.conjugate(D)
"""


def complex_D_wigner(l, a, b, c):

    d = wigner_d_matrix(l, b, dtype=np.complex64)
    # ea = diag_exp(l, a)
    # ec = diag_exp(l, c)
    # D = np.matmul(np.matmul(ea, d), ec)
    D = d
    # print(d)
    for p in range(2*l+1):
        for q in range(2*l+1):
            # D[q, p] *= np.exp(-(p-l)*1j*a)*np.exp(-(q-l)*1j*c)
            D[p, q] *= np.exp(-(p - l) * 1j * a) * np.exp(-(q - l) * 1j * c)
    # np.conjugate(D)
    # print(D)
    # D = np.flip(D, axis=0)
    # D = np.flip(D, axis=1)
    # D = np.conjugate(D)
    return D

def real_D_wigner(l, a, b, c):
    C = complex_to_real_sh(l)
    D = complex_D_wigner(l, a, b, c)
    D = np.real(C*D*np.conjugate(C.T))

    """
    Da = complex_D_wigner(l, a, 0., 0.)
    Da = np.real(C * Da * np.conjugate(C.T))
    Db = complex_D_wigner(l, 0., b, 0.)
    Db = np.real(C * Db * np.conjugate(C.T))
    Dc = complex_D_wigner(l, 0., 0., c)
    Dc = np.real(C * Dc * np.conjugate(C.T))
    """


    # return np.conjugate(C.T)*D*C
    # return np.real(C*D*np.conjugate(C.T))
    # return np.real(Da*Db*Dc)
    return D


"""
return a list of real wigner matrices D^l for l in [|0, l_max|]
"""
def real_D_wigner_from_euler(l_max, a, b, c):
    D = np.zeros(((l_max+1)**2, (l_max+1)**2))
    k = 0
    for l in range(l_max+1):
        D[k:k+(2*l+1), k:k+(2*l+1)] = real_D_wigner(l, a, b, c)
        k += 2*l+1
    return D

"""
return a list of real wigner matrices D^l for l in [|0, l_max|]
parametrized by a quaternion q
"""
def real_D_wigner_from_quaternion(l_max, q):
    r = Rotation(q)
    euler = r.as_euler('zyz')
    return real_D_wigner_from_euler(l_max, euler[0], euler[1], euler[2])

# to debug
"""
def wigner_d_matrix_(l, k1, k2, b):
    k = np.min([l + k2, l - k2, l + k1, l - k1])
    a = 0
    lbd = 0
    if k == l + k2:
        a = k1 - k2
        lbd = k1 - k2
    if k == l - k2:
        a = k2 - k1
        lbd = 0
    if k == l + k1:
        a = k2 - k1
        lbd = 0
    if k == l - k1:
        a = k1 - k2
        lbd = k1 - k2

    s_ = np.sin(b/2.)
    c_ = np.sin(b/2.)
    c = np.cos(b)

    b = 2*(l-k)-a
    d = scipy.special.jacobi(a, b, k)(c)
    d *= (s_**a)*(c_**b)
    d *= np.sqrt(scipy.special.binom(2*l-k, k+a)/scipy.special.binom(k+b, b))
    d *= (-1)**lbd

    return d

def wigner_d_matrix(l, b, dtype=np.float32):
    d = np.zeros(shape=(2*l+1, 2*l+1), dtype=dtype)
    for k1 in range(-l, l+1):
        for k2 in range(-l, l+1):
            d[k1+l, k2+l] = wigner_d_matrix_(l, k1, k2, b)
    return d

def wigner_D_matrix(l, a, b, c):
    D = np.asmatrix(wigner_d_matrix(l, b, dtype=np.complex64))
    for k1 in range(-l, l+1):
        D[k1, :] = np.exp(-k1*1j*a)*D[k1, :]
    for k2 in range(-l, l+1):
        D[:, k2] = np.exp(-k2 * 1j * c) * D[:, k2]
    return D
"""
"""
def unit_test3():
    angles = np.random.rand(3)
    # angles[0] = 0.0
    # angles[1] = 0.0
    # angles[2] = angles[0]
    # angles = [0., np.pi/2., 0.]
    # angles = [0., 1., 0.]
    R = euler_rot_zyz(angles[0], angles[1], angles[2])
    print(R)
    X = np.asmatrix(np.random.rand(1, 3))
    X_rot = (np.matmul(R, X.T)).T
    D = complex_wigner_2_(angles[0], angles[1], angles[2])
    # D = wigner_D_matrix(2, angles[0], angles[1], angles[2])

    print(angles[0])
    print(angles[1])
    print(np.exp(1j*angles[2]))
    print(D*D.T)

    print('orthogonality')
    print(np.linalg.norm(D * (D.conjugate()).T-np.eye(5)))
    # print(np.linalg.norm(D_bis * (D_bis.conjugate()).T - np.eye(5)))

    sh = complex_sh_(2, X)
    sh_rot = complex_sh_(2, X_rot)
    D_inv = (D.conjugate()).T
    y = np.matmul(D, sh.T) - sh_rot.T
    print(np.linalg.norm(y))

    print(np.real(y))
    print(np.real(sh-sh_rot))
    print(np.imag(y))
    print(np.imag(sh - sh_rot))

    # print(complex_sh_2_(np.asmatrix([[0., 1., 0.]])))
    # print(sh_rot)
    # print(np.sqrt(np.multiply(np.real(y), np.real(y)) + np.multiply(np.imag(y), np.imag(y))))
    # print(y)
    # print(D+D.conjugate())
"""

def complex_wigner_matrix_unit_test_(l, a, b, c, X):
    # angles = np.random.rand(3)
    # a = angles[0]
    # b = angles[1]
    # c = angles[2]

    D = complex_D_wigner(l, a, b, c)

    # D_ = complex_wigner_(l, a, b, c)
    # D_ = np.conjugate(D_.T)



    # print('u')
    # print(np.linalg.norm(D-D_))



    R = euler_rot_zyz(a, b, c)
    X_rot = (np.matmul(R.T, X.T)).T

    Y = complex_sh_(l, X)
    Y_rot = complex_sh_(l, X_rot)

    y = np.matmul(D, Y) - Y_rot
    # print(np.linalg.norm(Y))
    print(np.linalg.norm(y))


def complex_wigner_matrix_unit_test():
    for i in range(10):
        angles = np.random.rand(3)
        a = angles[0]
        b = angles[1]
        c = angles[2]

        # a = 0.
        # b = 0.
        # c = 0.

        X = np.asmatrix(np.random.rand(1, 3))
        print('wigner D test ', i)
        for l in range(4):
            print('l = ', l)
            complex_wigner_matrix_unit_test_(l, a, b, c, X)

def real_wigner_matrix_unit_test_(l, a, b, c, X):
    # angles = np.random.rand(3)
    # a = angles[0]
    # b = angles[1]
    # c = angles[2]

    D = real_D_wigner(l, a, b, c)

    # D_ = complex_wigner_(l, a, b, c)
    # D_ = np.conjugate(D_.T)



    # print('u')
    # print(np.linalg.norm(D-D_))



    R = euler_rot_zyz(a, b, c)
    X_rot = (np.matmul(R.T, X.T)).T

    Y = real_sh_(l, X)
    Y_rot = real_sh_(l, X_rot)

    y = np.matmul(D, Y) - Y_rot
    # print(np.linalg.norm(Y))
    print(np.linalg.norm(y))


def real_wigner_matrix_unit_test():
    for i in range(10):
        angles = np.random.rand(3)
        a = angles[0]
        b = angles[1]
        c = angles[2]

        # a = 0.
        # b = 0.
        # c = 0.

        X = np.asmatrix(np.random.rand(1, 3))
        print('wigner D test ', i)
        for l in range(4):
            print('l = ', l)
            real_wigner_matrix_unit_test_(l, a, b, c, X)




