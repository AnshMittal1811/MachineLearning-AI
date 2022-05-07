import numpy as np
import torch
from spherical_harmonics.wigner_matrix import complex_wigner_, complex_D_wigner, real_D_wigner, euler_rot_zyz
from spherical_harmonics.wigner_matrix import complex_to_real_sh, real_to_complex_sh

import scipy
from scipy import linalg, matrix, special

"""
Clebsch-Gordan coefficients allows to decompose tensor products of irreducible 
representations of SO(3)

https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients#_j2=0

⟨ j1 m1 j2 m2 | j3 m3 ⟩
⟨ j1,j2; m1,m2 | j1,j2; J,M ⟩
j1 = j1, j2 = j2, m1 = m1, m2 = m2, j3 = J, m3 = M

symmetries:
⟨ j1 m1 j2 m2 | j3 m3 ⟩ = (-1)**(j3-j1-j2)*⟨ j1 -m1 j2 -m2 | j3 -m3 ⟩
⟨ j1 m1 j2 m2 | j3 m3 ⟩ = (-1)**(j3-j1-j2)*⟨ j2 m1 j1 m2 | j3 m3 ⟩

when j2 = 0 the Clebsch–Gordan coefficients are given by deltaj3j1*deltam3m1
"""

"""
computes ⟨ j1 m1 j2 m2 | J M ⟩ for j1, m1, j2, m2 >= 0
"""
def clebsch_gordan_(j1, j2, J, m1, m2, M):
    # d = float((M == m1 + m2))
    if M != m1 + m2:
        return 0.0

    A = float((2*J+1)*np.math.factorial(J+j1-j2)*np.math.factorial(J-j1+j2)*np.math.factorial(j1+j2-J))
    A /= np.math.factorial(J+j1+j2+1)

    B = float(np.math.factorial(J+M)*np.math.factorial(J-M)*np.math.factorial(j1-m1)*
              np.math.factorial(j1+m1)*np.math.factorial(j2-m2)*np.math.factorial(j2+m2))
    C = 0.

    b0 = (j1+j2-J)
    b1 = (j1-m1)
    b2 = (j2+m2)

    a0 = 0
    a1 = (J-j2+m1)
    a2 = (J-j1-m2)

    k2 = np.min([b0, b1, b2])
    k1 = np.max([-a0, -a1, -a2])

    for k in range(k1, k2+1):
        a0_ = np.math.factorial(k+a0)
        a1_ = np.math.factorial(k+a1)
        a2_ = np.math.factorial(k+a2)

        b0_ = np.math.factorial(b0-k)
        b1_ = np.math.factorial(b1-k)
        b2_ = np.math.factorial(b2-k)

        C += ((-1)**k)/(float(a0_*a1_*a2_*b0_*b1_*b2_))

    return np.sqrt(A * B) * C

"""
computes ⟨ j1 m1 j2 m2 | J M ⟩
"""
def clebsch_gordan_coeff(j1, j2, J, m1, m2, M):
    if M < 0:
        if j1 >= j2:
            return (-1.)**(J-j1-j2)*clebsch_gordan_(j1, j2, J, -m1, -m2, -M)
        else:
            return clebsch_gordan_(j2, j1, J, -m2, -m1, -M)
    else:
        if j1 >= j2:
            return clebsch_gordan_(j1, j2, J, m1, m2, M)
        else:
            return (-1.) ** (J - j1 - j2) * clebsch_gordan_(j2, j1, J, m2, m1, M)


"""
computes the projection from type (j1, j2) to type J 
Q*kron(Dj1, Dj2)*Q.T = DJ
"""
def np_real_clebsch_gordan_projector(j1, j2, J, matrix_shape=True, dtype=np.float32):
    #Q = np.zeros(shape=(2 * J + 1, (2 * j1 + 1) * (2 * j2 + 1)), dtype=dtype)
    Q = np.zeros(shape=(2 * J + 1, 2 * j1 + 1, 2 * j2 + 1), dtype=dtype)
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            m3 = m1 + m2
            if -J <= m3 <= J:
                #Q[m3 + J, (2 * j2 + 1) * (m1 + j1) + (m2 + j2)] = clebsch_gordan_coeff(j1, j2, J, m1, m2, m3)
                Q[m3 + J, m1 + j1, m2 + j2] = clebsch_gordan_coeff(j1, j2, J, m1, m2, m3)
    Q = np.reshape(Q, newshape=(2*J+1, -1))
    CRj1 = complex_to_real_sh(j1)
    RCj1 = np.conjugate(CRj1.T)

    CRj2 = complex_to_real_sh(j2)
    RCj2 = np.conjugate(CRj2.T)

    CRJ = complex_to_real_sh(J)
    # RCJ = np.conjugate(CRJ.T)
    Q = np.matmul(np.matmul(CRJ, Q), np.kron(RCj1, RCj2))
    Q = np.real(Q)
    Q = Q.astype(dtype=dtype)
    if not matrix_shape:
        Q = np.reshape(Q, newshape=(2*J+1, 2*j1+1, 2*j2+1))
    return Q

def sparse_matrix(M, eps=0.00001):
    m = np.shape(M)[0]
    n = np.shape(M)[1]
    idx_ = []
    coeffs_ = []
    n_ = 0
    for i in range(m):
        idx_.append([])
        coeffs_.append([])
        for j in range(n):
            if np.abs(M[i, j]) > eps:
                idx_[-1].append(j)
                coeffs_[-1].append(M[i, j])
        n_ = max(n_, len(idx_[-1]))

    idx = np.zeros((m, n_), dtype=np.int32)
    coeffs = np.zeros((m, n_), dtype=np.float32)
    for i in range(m):
        for j in range(len(idx_[i])):
            idx[i, j] = idx_[i][j]
            coeffs[i, j] = coeffs_[i][j]

    return coeffs, idx

"""
Computes the Clebsch Gordan decomposition 
"""
def np_clebsch_gordan_decomposition(j1, j2, matrix_shape=True, l_max=None, dtype=np.float32):
    Q = []
    for J in range(abs(j1-j2), min(j1+j2+1, l_max+1)):
        Q.append(np_real_clebsch_gordan_projector(j1, j2, J, matrix_shape=matrix_shape, dtype=dtype))
    Q = np.concatenate(Q, axis=0)
    return Q

def torch_clebsch_gordan_decomposition_(j1, j2, sparse=False, l_max=None, dtype=torch.float32):
    Q = np_clebsch_gordan_decomposition(j1, j2, matrix_shape=True, l_max=l_max, dtype=np.float32)
    if sparse:
        coeffs, idx = sparse_matrix(Q)
        idx = torch.from_numpy(idx).to(torch.int64)
        coeffs = torch.from_numpy(coeffs).to(torch.float32)
        return coeffs, idx
    else:
        return torch.from_numpy(Q).to(dtype)

def representation_type(x):
    j1 = int((x.shape[-3] - 1) / 2)
    j2 = int((x.shape[-2] - 1) / 2)
    return (j1, j2)

def decompose_(x, Q):
    j1, j2 = representation_type(x)
    # Q = Q[(j1, j2)]
    s = list(x.shape)
    c = s[-1]
    s.pop()
    s[-2] = -1
    s[-1] = c
    x = torch.reshape(x, shape=s)
    x = torch.einsum('ij,...jk->...ik', Q.type_as(x), x)
    y = []
    p = 0
    for J in range(abs(j1-j2), j1+j2+1):
        y.append(x[..., p:p+2*J+1, :])
        p += 2*J+1
    return y

def sparse_decompose_(x, coeffs, idx):
    j1, j2 = representation_type(x)
    # coeffs = Q[(j1, j2)][0]
    # idx = Q[(j1, j2)][0]
    s = list(x.shape)
    c = s[-1]
    s.pop()
    s[-2] = -1
    s[-1] = c
    x = torch.reshape(x, shape=s)

    x = (x[..., idx, :])
    x = torch.einsum('ij,...ijk->...ik', coeffs, x)
    y = []
    p = 0
    for J in range(abs(j1-j2), j1+j2+1):
        y.append(x[..., p:p+2*J+1, :])
        p += 2*J+1
    return y


class torch_clebsch_gordan_decomposition:
    def __init__(self, l_max, l_max_out=None, sparse=False, output_type='dict'):
        self.dtype = torch.float32
        self.l_max = l_max
        self.sparse = sparse
        self.output_type = output_type
        if l_max_out is None:
            self.l_max_out = 2*l_max
        else:
            self.l_max_out = l_max_out
        self.Q = dict()

        for j1 in range(self.l_max+1):
            for j2 in range(self.l_max+1):
                if self.l_max_out < abs(j1-j2):
                    pass
                elif self.sparse:
                    coeffs, idx = torch_clebsch_gordan_decomposition_(j1, j2, l_max=self.l_max_out,
                                                                   sparse=True, dtype=self.dtype)
                    self.Q[(j1, j2)] = [coeffs, idx]
                else:
                    self.Q[(j1, j2)] = torch_clebsch_gordan_decomposition_(j1, j2, l_max=self.l_max_out,
                                                                        sparse=False, dtype=self.dtype)

    def decompose(self, x):
        if not isinstance(x, list):
            x = [x]
        y = dict()
        for i in range(len(x)):
            j1, j2 = representation_type(x[i])
            if self.sparse:
                yi = sparse_decompose_(x[i], self.Q[(j1, j2)][0], self.Q[(j1, j2)][1])
            else:
                yi = decompose_(x[i], self.Q[(j1, j2)])
            for J in range(abs(j1-j2), min(j1+j2+1, self.l_max_out+1)):
                if not str(J) in y:
                    y[str(J)] = []
                y[str(J)].append(yi[J-abs(j1-j2)])

        for J in y:
            y[J] = torch.cat(y[J], dim=-1)

        if self.output_type == 'list':
            return y.values()
        else:
            return y

"""
computes the Clebsch Gordan decomposition of the Zernike basis (up to degree d) tensored by a dimension 2*k+1
equivariant feature.
"""
"""
def np_zernike_clebsch_gordan_decomposition(d, k, matrix_shape=True, l_max=None, dtype=np.float32):
    zerinke_basis_idx = []
    size_in = 0
    size_out = 0
    num_out_features = [0]*(d+1+k+1)
    for l in range(1, d + 1):
        for n in range(min(2 * d - l + 1, l + 1)):
            if (n - l) % 2 == 0:
                size_in += 2*l + 1
                zerinke_basis_idx.append((n, l))
                
                np_clebsch_gordan_decomposition(l, k, matrix_shape=True, l_max=l_max, dtype=np.float32)
                for J in range(abs(l-k), min(l+k+1, l_max+1)):
                    num_out_features[J] += 1
                    size_out


    Q = np.zeros()

    for i in range(len())

    for J in range(abs(j1-j2), min(j1+j2+1, l_max+1)):
        Q.append(np_real_clebsch_gordan_projector(j1, j2, J, matrix_shape=matrix_shape, dtype=dtype))
    Q = np.concatenate(Q, axis=0)
    return Q

"""


def real_conj(A, Q):
    return np.matmul(Q.T, np.matmul(A, Q))

def complex_conj(A, Q):
    return np.matmul(np.conjugate(Q.T), np.matmul(A, Q))

def unit_test4():

    j1 = 2
    j2 = 2
    J = 2
    # cb_dict = clebsch_gordan_dict()

    Q = np.asmatrix(clebsch_gordan_matrix(j1, j2, J, dtype=np.complex64))
    # Q = np.sqrt(2.) * Q

    # Q_ = np.asmatrix(Q_from_cb_dict(j1, j2, J, cb_dict, dtype=np.complex64))
    # Q_ = np.sqrt(2.)*Q_

    # Q__ = tensorProductDecompose_(j1, j2, J)
    # Q__ = np.sqrt(1./2.49634557e-02)*Q__

    angles = np.random.rand(3)
    # angles = [1., 0., 0.]


    Dj1 = complex_wigner_(j1, angles[0], angles[1], angles[2])
    Dj2 = complex_wigner_(j2, angles[0], angles[1], angles[2])
    DJ = complex_wigner_(J, angles[0], angles[1], angles[2])



    print('eee')

    prod = np.kron(Dj1, Dj2)

    y = np.matmul(np.matmul(Q, prod), Q.T) - DJ

    # print(y)
    # print(np.matmul(Q.T, Q))
    # print(np.matmul(Q, Q.T))

    # print(np.real(prod))
    # print(np.real(Q))
    # print(np.real(Q__))
    # y = np.matmul(Q, prod) - np.matmul(DJ, Q)
    # y = np.matmul(y, Q.T)
    # print(np.linalg.norm(Q - Q_, 'fro'))
    print(np.linalg.norm(y))
    print(np.linalg.norm(DJ))




def unit_test5():
    j1 = 1
    j2 = 1

    angles = np.random.rand(3)
    # angles = [1.0, 0.0, 0.]

    D0 = np.asmatrix([[1.]], dtype=np.complex64)
    D1 = complex_wigner_(1, angles[0], angles[1], angles[2])
    D2 = complex_wigner_(2, angles[0], angles[1], angles[2])

    D = [D0, D1, D2]

    prod = np.kron(D[j1], D[j2])
    # prod = np.kron(D[j2], D[j1])

    c = 0.0
    for m1 in range(-j1, j1+1):
        for k1 in range(-j1, j1+1):
            for m2 in range(-j2, j2+1):
                for k2 in range(-j2, j2+1):
                    a = D[j1][j1 + m1, j1 + k1] * D[j2][j2 + m2, j2 + k2]
                    b = 0.
                    # b = prod[(2*j2+1)*(m1+j1) + (m2+j2), (2*j2+1)*(k1+j1) + (k2+j2)]

                    for J in range(abs(j1-j2), j1+j2+1):
                        if(2*J >= m1+m2+J >= 0 and 2*J >= k1+k2+J >= 0):
                            b += D[J][m1+m2+J, k1+k2+J] * clebsch_gordan_coeff(j1, j2, J, m1, m2, m1 + m2) * clebsch_gordan_coeff(j1, j2, J, k1, k2, k1 + k2)

                    print('zz')
                    print(a)
                    print(b)
                    print(a-b)

                    c += abs(np.real(a-b))*abs(np.real(a-b))+abs(np.imag(a-b))*abs(np.imag(a-b))

    print('rr')
    print(np.sqrt(c))



def unit_test6():
    angles = np.random.rand(3)
    # angles = [0., 1., 0.]

    Q0 = np.asmatrix(clebsch_gordan_matrix(1, 1, 0, dtype=np.complex64))
    Q1 = np.asmatrix(clebsch_gordan_matrix(1, 1, 1, dtype=np.complex64))
    Q2 = np.asmatrix(clebsch_gordan_matrix(1, 1, 2, dtype=np.complex64))

    D0 = np.asmatrix([[1.]], dtype=np.complex64)
    D1 = complex_wigner_(1, angles[0], angles[1], angles[2])
    D2 = complex_wigner_(2, angles[0], angles[1], angles[2])

    y = np.kron(D1, D1) - real_conj(D2, Q2) - real_conj(D1, Q1) - real_conj(D0, Q0)

    print(np.linalg.norm(y))

def tensor_decomposition_unit_test___(j, k, J, a, b, c):
    Dj = complex_D_wigner(j, a, b, c)
    Dk = complex_D_wigner(k, a, b, c)
    DJ = complex_D_wigner(k, a, b, c)
    assert(j+k >= J >= abs(k-j))
    QJ = clebsch_gordan_matrix(j, k, J)

    y = real_conj(np.kron(Dj, Dk), QJ.T) - DJ
    print(np.linalg.norm(y))


def tensor_decomposition_unit_test__(j, k, a, b, c):
    Dj = complex_D_wigner(j, a, b, c)
    Dk = complex_D_wigner(k, a, b, c)

    D_ = np.zeros(shape=((2*j+1)*(2*k+1), (2*j+1)*(2*k+1)), dtype=np.complex64)
    D = np.kron(Dj, Dk)

    for J in range(abs(k-j), k+j+1):
        print('j = ', j, 'k = ', k, 'J = ', J)
        DJ = complex_D_wigner(J, a, b, c)
        QJ = clebsch_gordan_matrix(j, k, J)
        y = real_conj(np.kron(Dj, Dk), QJ.T) - DJ
        # print(np.linalg.norm(DJ))
        print(np.linalg.norm(y))
        D_ += real_conj(DJ, QJ)
    print('decompose j = ', j, 'k = ', k)
    print(np.linalg.norm(D - D_))


def tensor_decomposition_unit_test(l):
    for i in range(10):
        angles = np.random.rand(3)
        a = angles[0]
        b = angles[1]
        c = angles[2]
        for j in range(l+1):
            for k in range(l+1):
                tensor_decomposition_unit_test__(j, k, a, b, c)

def invariant_feature(equivariant_features, p, q, Q):
    # y = tf.einsum('bvqmrc,bvqnrc->bvqmnrc', equivariant_features[p[0]], equivariant_features[p[1]])
    # the equivariant channels must in the last dimesion

    #y = tf.einsum('bvqrcm,bvqrcn->bvqrcmn', equivariant_features[p[0]], equivariant_features[p[1]])
    """
    nb = y.get_shape()[0].value
    nv = y.get_shape()[1].value
    nq = y.get_shape()[2].value
    nr = y.get_shape()[3].value
    nc = y.get_shape()[4].value
    y = tf.reshape(y, shape=(nb, nv, nq, nr, nc, -1))
    """

def higher_product_matrix(p, q):


    Q = npClebschGordanMatrices(3)

    """
    res = np.eye((2*abs(q[0])+1)*(2*abs(p[1])+1))
    I = np.eye(1)
    res = np.real(np.reshape(Q.getMatrix(q[0], p[1], q[1]), newshape=(2*abs(q[1])+1, -1)))
    for i in range(len(p)-1):


        Qi_ = np.real(np.reshape(Q.getMatrix(q[i+1], p[i+2], q[i+2]), newshape=(2*abs(q[i+1])+1, -1)))
        Qi = np.kron(Qi_, I)
        res = np.matmul(Qi_, np.kron(res, I))
        I = np.kron(I, np.eye(2 * abs(p[i + 1]) + 1))
    """


    Q1 = np.reshape(Q.getMatrix(q[0], p[1], q[1]), newshape=(2*abs(q[1])+1, -1))
    Q2 = np.reshape(Q.getMatrix(q[1], p[2], q[2]), newshape=(2 * abs(q[2]) + 1, -1))
    M = np.real(Q1)
    M = np.kron(M, np.eye(2*abs(p[2])+1))
    M = np.matmul(np.real(Q2), M)
    print(M)
    print(np.matmul(M, M.transpose()))
    return


def higher_product(R, X, p, q, Q):

    # print(np.linalg.norm(y))
    # print(y)
    X = np.asmatrix(np.random.rand(1, 3))
    X /= (np.linalg.norm(X))
    X *= 10.0
    X_rot = (np.matmul(R.T, X.T)).T

    y = complex_sh_(abs(p[0]), X)
    y_rot = complex_sh_(abs(p[0]), X_rot)
    for i in range(len(p)-1):
        """
        print('uuu')
        print(X.shape)
        print(y.shape)
        print(p)
        print(complex_sh_(abs(p[i+1]), X).shape)
        print(Q.getMatrix(q[i], p[i+1], q[i+1]).shape)
        print('aaa')
        """
        """
        print('aaaaaa')
        print(q[i], p[i+1], q[i+1])
        print(Q.getMatrix(q[i], p[i+1], q[i+1]))
        print('bbbbbb')
        """
        X = np.asmatrix(np.random.rand(1, 3))
        X /= (np.linalg.norm(X))
        X *= 10.0
        X_rot = (np.matmul(R.T, X.T)).T


        z = np.einsum('jmn,jmn->j', Q.getMatrix(q[i], p[i+1], q[i+1]), Q.getMatrix(q[i], p[i+1], q[i+1]))
        print('qi, pi+1, qi+1 = ', q[i], p[i + 1], q[i + 1])
        print('norm z= ', np.linalg.norm(z))
        y = np.einsum('vm,vn->vmn', y, complex_sh_(abs(p[i+1]), X))
        y_rot = np.einsum('vm,vn->vmn', y_rot, complex_sh_(abs(p[i+1]), X_rot))
        y = np.einsum('jmn,vmn->vj', Q.getMatrix(q[i], p[i+1], q[i+1]), y)
        y_rot = np.einsum('jmn,vmn->vj', Q.getMatrix(q[i], p[i + 1], q[i + 1]), y_rot)
        # print(y)
        # print(np.linalg.norm(y))

    return y

def higher_tensor_decomposition_unit_test():
    Q = npClebschGordanMatrices(3)
    p = []
    q = []

    # degree 1 invariants
    p.append(np.zeros(shape=(1, 1), dtype=np.int32))
    q.append(np.zeros(shape=(1, 1), dtype=np.int32))

    p.append(np.array([[1, 1], [2, 2]], dtype=np.int32))
    q.append(np.array([[1, 0], [2, 0]], dtype=np.int32))

    p.append(np.array([[1, 1, 1],
                       [1, 1, 2],
                       [1, 2, 2],
                       [2, 2, 2]], dtype=np.int32))
    q.append(np.array([[1, 1, 0],
                       [1, 2, 0],
                       [1, 2, 0],
                       [2, 2, 0]], dtype=np.int32))
    for i in range(10):
        angles = np.random.rand(3)
        X = np.asmatrix(np.random.rand(1, 3))
        X /= (np.linalg.norm(X))
        a = angles[0]
        b = angles[1]
        c = angles[2]
        R = euler_rot_zyz(a, b, c)
        for d in range(len(p)):
            for j in range(np.size(p[d], 0)):
                print(p[d][j, :])
                print(q[d][j, :])
                z = higher_product(R, X, p[d][j, :], q[d][j, :], Q)
                print('norm output = ', np.linalg.norm(z))
                # print(np.linalg.norm(X))

def real_tensor_decomposition_unit_test__(j, k, a, b, c):

    # CRj = complex_to_real_sh(j)
    # CRk = complex_to_real_sh(k)

    # K = np.kron(CRj, CRk)
    # K_T = np.conjugate(K.T)

    Dj = real_D_wigner(j, a, b, c)
    Dk = real_D_wigner(k, a, b, c)

    D_ = np.zeros(shape=((2*j+1)*(2*k+1), (2*j+1)*(2*k+1)), dtype=np.complex64)
    D = np.kron(Dj, Dk)

    for J in range(abs(k-j), k+j+1):
        print('j = ', j, 'k = ', k, 'J = ', J)
        # CRJ = complex_to_real_sh(J)
        # RCJ = np.conjugate(CRJ.T)
        DJ = real_D_wigner(J, a, b, c)
        QJ = real_Q_from_cb(j, k, J, dtype=np.complex64)

        y = complex_conj(np.kron(Dj, Dk), np.conjugate(QJ.T)) - DJ
        # print(np.linalg.norm(DJ))
        print(np.linalg.norm(y))
        D_ += real_conj(DJ, QJ)
    print('decompose j = ', j, 'k = ', k)
    print(np.linalg.norm(D - D_))


def real_tensor_decomposition_unit_test(l):
    for i in range(3):
        angles = np.random.rand(3)
        a = angles[0]
        b = angles[1]
        c = angles[2]
        for j in range(l+1):
            for k in range(l+1):
                real_tensor_decomposition_unit_test__(j, k, a, b, c)






