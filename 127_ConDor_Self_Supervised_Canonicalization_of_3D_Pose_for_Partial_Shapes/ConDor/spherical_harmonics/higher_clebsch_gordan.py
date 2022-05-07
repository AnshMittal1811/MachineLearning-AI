import numpy as np
import ast
from clebsch_gordan_decomposition import npClebschGordanMatrices

from scipy.linalg import orth

"""
We have variables x_ij for i in [|0, nb_irreducible-1|] and j in [|-i,+i|] 
A higher tensor product of these consist of a tensor of the form
T = prod_k x_(i_k)(j_k) for i_k in [|0, nb_irreducible-1|] and j_k in [|-i_k, i_k|]
Given lists p and q of positive integers such that for all i q[i] in [|p[i+1]-p[i]|, p[i+1]+p[i]|]
we have a projection matrix Q^{p,q} sending T to RR^{2q[-1]+1} or (CC^{2q[-1]+1}) this class computes Q^{p,q}.

real_wigner: if true we use the real spherical harmonics basis otherwise we use the complex one
"""

class npHigherClebschGordan:
    def __init__(self, max_degree=3, real_wingner=False, dtype=np.complex64):
        self.max_degree = max_degree
        # self.is_matrix = is_matrix
        self.real_wigner = real_wingner
        self.dtype = dtype
        self.Q = npClebschGordanMatrices(max_degree, False, real_wingner, dtype)

    def getMatrix(self, p, q, as_matrix=True):
        Q = self.Q.getMatrix(q[0], p[1], q[1])
        # Q = np.transpose(Q, axes=(0, 2, 1))
        print('aaa')
        print(p)
        print(q)
        print('bbb')
        print(Q.shape)
        for i in range(len(p)-2):
            # Q = np.einsum('...i, ijk->...jk', Q, self.Q.getMatrix(q[i+1], p[i+1], q[i+2]))
            """
            print('trdhtrd')
            print(np.shape(Q))
            print(q[i + 1], p[i + 1], q[i + 2])
            print(np.shape(self.Q.getMatrix(q[i + 1], p[i + 2], q[i + 2])))
            """
            # Q = np.einsum('kij,i...->kj...', self.Q.getMatrix(q[i + 1], p[i + 2], q[i + 2]), Q)
            print(self.Q.getMatrix(q[i + 1], p[i + 2], q[i + 2]).shape)
            Q = np.einsum('jik,i...->j...k', self.Q.getMatrix(q[i + 1], p[i + 2], q[i + 2]), Q)

        if as_matrix:
            # Q = np.reshape(Q, newshape=(-1, q[-1]))
            # Q = np.transpose(Q, axes=(1, 0))

            Q = np.reshape(Q, newshape=(2*abs(q[-1])+1, -1))
        return Q

"""
def sh_to_global_index_conversion_(l, m):
    return np.multiply(l+1, l+1) + m - 1

def global_to_sh_index_conversion_(i):
    l = np.floor(np.sqrt(i+1))-1
    m = i + 1 - np.multiply(l+1, l+1)
    return [l, m]
"""

def sh_to_global_index_conversion_(l, m):
    return np.multiply(l, l) + m

def global_to_sh_index_conversion_(i):
    l = np.floor(np.sqrt(i))
    m = i - np.multiply(l, l)
    return [l, m]

def list_to_string_(L):
    return '[' + ', '.join(str(e) for e in L) + ']'

def string_to_list_(s):
    return ast.literal_eval(s)

def multi_index_(I, p, prod):
    idx = []
    d = len(p)
    # k = idx.size();
    for i in range(d):
        dimp = 2 * abs(p[d - 1 - i]) + 1
        idx_ = I % dimp
        idx = [idx_] + idx
        I -= idx_
        I /= dimp
    return idx

def monomial_index_(I, p, prod):
    idx = multi_index_(I, p, prod)
    for i in range(len(p)):
        idx[i] = sh_to_global_index_conversion_(int(abs(p[i])), int(idx[i]))
    idx.sort()
    return list_to_string_(idx)

"""
We have variables x_ij for i in [|0, nb_irreducible-1|] and j in [|-i,+i|] 
we sort them in the following way:
y_0 := x_00, y_1 = x_1-1, y_2 = x_10, y_3 = x_11, y_4 = x_2-2, y_5 = x_2-1 ...
A tensor product of these consist of a tensor of the form
T = prod_k x_(i_k)(j_k) for i_k in [|0, nb_irreducible-1|] and j_k in [|-i_k, i_k|]
Given lists p and q of positive integers such that for all i q[i] in [|p[i+1]-p[i]|, p[i+1]+p[i]|]
we have a projection matrix Q^{p,q} sending T to RR^{2q[-1]+1} or (CC^{2q[-1]+1}).
The coefficients of Q^{p,q}T are polynomials in the variables y_k
The following class computes these polynomials.
we return polynomials as a list of monomials and a list of the associated coefficients 
each monomial is a list of indices for instance y_1y_3y_8 would be represented by the list [1,3,8]

real_wigner: if true we use the real spherical harmonics basis otherwise we use the complex one
"""

class npClebschGordanPolynomial:
    def __init__(self, nb_irreducible=3, real_wingner=False, dtype=np.complex64):

        self.nb_irreducible = nb_irreducible
        self.real_wigner = real_wingner
        self.dtype = dtype
        self.Q = npHigherClebschGordan(nb_irreducible, real_wingner, dtype)
    def getPolynomial(self, p, q, eps=0.00001):
        d = len(p)
        Q = self.Q.getMatrix(p, q, as_matrix=True)


        print('-------------------------------')
        print('p: ', p)
        print('q: ', q)
        print(Q)
        print('-------------------------------')

        m = 2*abs(q[d-1])+1
        n = int(np.prod(2*np.absolute(p)+1))
        idx = np.nonzero(Q)
        # idx[0] : 'row indices'
        # idx[1] : 'col indices'
        nb_coeffs = len(idx[0])
        values = []
        coeffs_ = []
        for i in range(m):
            coeffs_i = dict()
            for j in range(n):
                idx_ij = monomial_index_(j, p, n)

                if idx_ij in coeffs_i and abs(Q[i, j]) > eps:
                    coeffs_i[idx_ij] += Q[i, j]
                elif abs(Q[i, j]) > eps:
                    coeffs_i[idx_ij] = Q[i, j]
            to_remove = []
            for key in coeffs_i:
                if abs(coeffs_i[key]) < eps:
                    to_remove.append(key)
            for key in to_remove:
                coeffs_i.pop(key)
            # if len(coeffs_i) > 0:
            coeffs_.append(coeffs_i.copy())
        return coeffs_




def list_increasing_indices_recursive__(N, p, j, idx):
    d = len(p)
    if j >= d - 1:
        idx.append(np.copy(p))
        return
    k1 = max(p[j], 1)
    for p_next in range(k1,N+1):
        p[j+1] = p_next
        list_increasing_indices_recursive__(N, p, j+1, idx)




def list_increasing_indices_recursive_(N, d, idx, start_at_1 = True):
    p = np.zeros(shape=(d,), dtype=np.int32)
    p0_ = 0
    if start_at_1:
        p0_ = 1
    for p0 in range(p0_, N+1):
        p[0] = p0
        list_increasing_indices_recursive__(N, p, 0, idx)


def list_increasing_indices_recursive(N, d_max):
    res = []
    for d in range(2, d_max+1):
        idx = []
        list_increasing_indices_recursive_(N, d, idx)
        # res += list_increasing_indices_recursive_(N, d, idx)
        # res.append(idx)
        res += idx
        # res.append(list_increasing_indices_recursive_(N, d, idx))

    return res



def list_irreducible_invariants_recursive_q_(p, q, i, idx, order_bound=100):
    d = len(p)
    k1 = abs(abs(p[i+1])-abs(q[i]))
    k2 = min(abs(p[i+1])+abs(q[i]), order_bound)
    if i == d - 2:
        if k1 == 0:
            q[d-1] = 0
            idx.append(np.copy(q))
        return

    k1 = max(k1, 1)
    for q_next in range(k1, k2+1):
        q[i + 1] = q_next
        list_irreducible_invariants_recursive_q_(p, q, i+1, idx, order_bound)

def list_irreductible_invariants_recursive_q(p, order_bound=100):
    d = len(p)
    q = np.zeros(shape=(d,), dtype=np.int32)
    q[0] = p[0]
    idx = []
    list_irreducible_invariants_recursive_q_(p, q, 0, idx, order_bound)
    return idx


def list_irreducible_invariants_recursive(N, max_deg, order_bound=100):
    idx = [None]*(max_deg)
    p = list_increasing_indices_recursive(N, max_deg)
    for i in range(len(p)):
        q = list_irreductible_invariants_recursive_q(p[i], order_bound=order_bound)
        if idx[len(p[i])-1] is None:
            idx[len(p[i])-1] = [[np.copy(p[i]), q.copy()]]
        else:
            idx[len(p[i]) - 1].append([np.copy(p[i]), q.copy()])


    p0 = np.array([0], dtype=np.int32)
    l0 = [p0]

    idx[0] = [[p0, l0]]

    n_deg_d_invar = 0
    n_invar = 0
    for d in range(1, max_deg+1):
        n_deg_d_invar = 0
        for i in range(len(idx[d-1])):
            n_deg_d_invar += len(idx[d-1][i][1])
        print('nb deg ', d , ' invariants: ', n_deg_d_invar)
        n_invar += n_deg_d_invar

    dim_quotient = (N + 1) * (N + 1) - 3
    print('N= ', N)
    print('max deg= ', max_deg)
    print('order bound= ', order_bound)
    print('nb invariants= ', n_invar)
    print('dim quotient= ', dim_quotient)
    print('nb missing invars= ',  dim_quotient - n_invar)

    return idx

"""
orthonormalize a family of polynomials in a monomial basis.
"""
def orthonormalize_polynomials(monomial_basis, polynomials, eps, dtype=np.float32):
    # build matrix
    nb_monomials = len(monomial_basis)
    nb_polynomials = len(polynomials)
    A = np.zeros(shape=(nb_monomials, nb_polynomials), dtype=dtype)
    for i in range(nb_polynomials):
        for key in polynomials[i]:
            j = monomial_basis.index(key)
            A[j, i] = polynomials[i][key]
    # orthonormalize
    """
    print('+++++++++++++++++++++++++')
    print(A)
    print('+++++++++++++++++++++++++')
    """
    Q = orth(A)
    """
    print('*************************')
    print(Q)
    print('*************************')
    """

    nb_polynomials = Q.shape[-1]
    new_monomial_basis = set()
    new_polynomials = []
    for j in range(nb_polynomials):
        coeffs_j = dict()
        for i in range(nb_monomials):
            if abs(Q[i, j]) > eps:
                key = monomial_basis[i]
                new_monomial_basis.add(key)
                if key in coeffs_j:
                    coeffs_j[key] += Q[i, j]
                else:
                    coeffs_j[key] = Q[i, j]
        to_remove = []
        for key in coeffs_j:
            if abs(coeffs_j[key]) < eps:
                to_remove.append(key)
        for key in to_remove:
            coeffs_j.pop(key)
        if len(coeffs_j) > 0:
            new_polynomials.append(coeffs_j.copy())
    return list(new_monomial_basis), new_polynomials

"""
#------------ Compute irreducible SO(3) invariant polynomials ----------#
We compute a orthonormal basis of irreducible polynomials of degree less than max_degree for the wigner action:
we have variables x_ij for i in [|0, nb_irreducible-1|] and j in [|-i,+i|] 
we sort them in the following way:
y_0 := x_00, y_1 = x_1-1, y_2 = x_10, y_3 = x_11, y_4 = x_2-2, y_5 = x_2-1 ...
we return polynomials as a list of monomials and a list of the associated coefficients 
each monomial is a list of indices for instance y_1y_3y_8 would be represented by the list [1,3,8]

real_wigner: if true we use the real spherical harmonics basis otherwise we use the complex one
max_degree: maximal degree of the invariant polynomials computed
nb_irreducible: The number of irreducible SO( 3 ) representations we consider 
"""

class npInvariantPolynomials:
    def __init__(self, real_wigner, max_degree, nb_irreducible, dtype=None, max_order=100):
        # nb_irreducible -= 1
        self.real_wigner = real_wigner
        if dtype is None:
            self.dtype = np.float32
        if real_wigner:
            self.dtype = np.complex64
        self.max_degree = max_degree
        self.nb_irreducible = nb_irreducible
        self.max_order = max_order
        self.P = npClebschGordanPolynomial(nb_irreducible=nb_irreducible, real_wingner=real_wigner, dtype=dtype)
        self.idx = list_irreducible_invariants_recursive(nb_irreducible, max_degree, order_bound=100)
        self.monomials = []
        self.polynomials = []

        # monomials_tmp = []
        # polynomials_tmp = []


        self.monomial_idx = []
        self.monomial_coeff = []
        for d in range(1, len(self.idx)):
            deg_d_monomials = set()
            deg_d_polynomials = []
            # print('d = ', d)
            for p_idx in range(len(self.idx[d])):
                p = self.idx[d][p_idx][0]
                for q_idx in range(len(self.idx[d][p_idx][1])):
                    q = self.idx[d][p_idx][1][q_idx]
                    Ppq = self.P.getPolynomial(p, q)
                    if len(Ppq[0]) > 0:
                        deg_d_polynomials.append(Ppq[0].copy())
                        for key in Ppq[0]:
                            # deg_d_monomials.add(Ppq[0][key])
                            deg_d_monomials.add(key)
            # self.monomials.append(list(deg_d_monomials.copy()))
            # self.polynomials.append(deg_d_polynomials.copy())

            monomials_tmp, polynomials_tmp = orthonormalize_polynomials(list(deg_d_monomials),
                                                                        deg_d_polynomials,
                                                                        eps=0.00001, dtype=dtype)
            self.monomials.append(monomials_tmp.copy())
            self.polynomials.append(polynomials_tmp.copy())


        for d in range(len(self.polynomials)):
            deg_d_monomial_idx = []
            deg_d_monomial_coeff = []
            for i in range(len(self.polynomials[d])):
                deg_d_monomial_idx_pq = []
                deg_d_monomial_coeff_pq = []
                for key in self.polynomials[d][i]:
                    deg_d_monomial_idx_pq.append(self.monomials[d].index(key))
                    deg_d_monomial_coeff_pq.append(self.polynomials[d][i][key])
                deg_d_monomial_idx.append(np.array(deg_d_monomial_idx_pq, dtype=np.int32))
                deg_d_monomial_coeff.append(np.array(deg_d_monomial_coeff_pq, dtype=self.dtype))
            self.monomial_idx.append(deg_d_monomial_idx.copy())
            self.monomial_coeff.append(deg_d_monomial_coeff.copy())

        monomials = self.monomials.copy()
        self.monomials = []
        nb_polynomials = 1
        for d in range(len(self.polynomials)):
            print('-------------------------')
            print('degree ', d+2, ' monomials')
            print('nb deg ', d+2 ,' monomials= ', len(monomials[d]))
            print(monomials[d])
            print('-------------------------')

            print('-------------------------')
            print('degree ', d+2, ' polynomials')
            print('nb deg ', d+2,' polynomials= ', len(self.polynomials[d]))
            for i in range(len(self.polynomials[d])):
                print(self.polynomials[d][i])
            print('-------------------------')

            print('-------------------------')
            print('degree ', d+2, ' monomials idx')
            print('nb deg ', d+2, ' monomials idx= ', len(self.monomial_idx[d]))
            for i in range(len(self.monomial_idx[d])):
                print(self.monomial_idx[d][i])
            print('-------------------------')

            print('-------------------------')
            print('degree ', d+2, ' monomials coeffs')
            print('nb deg ', d+2, ' monomials coeffs= ', len(self.monomial_coeff[d]))
            for i in range(len(self.monomial_coeff[d])):
                print(self.monomial_coeff[d][i])
            print('-------------------------')

            nb_polynomials += len(self.polynomials[d])

            deg_d_monomials = np.zeros(shape=(d+2, len(monomials[d])), dtype=np.int32)
            for i in range(len(monomials[d])):
                idx = string_to_list_(monomials[d][i])
                for j in range(len(idx)):
                    deg_d_monomials[j, i] = idx[j]
            self.monomials.append(np.copy(deg_d_monomials.copy()))

        N = nb_irreducible
        dim_quotient = (N + 1) * (N + 1) - 3 # dimension of the quotient of the respresentation space by SO( 3 )
        print('L= ', N)
        print('max deg= ', max_degree)
        print('order bound= ', max_order)
        print('nb invariants= ', nb_polynomials)
        print('dim quotient= ', dim_quotient)
        print('nb missing invars= ', dim_quotient - nb_polynomials)

    def get_monomials(self, deg):
        return self.monomials[deg]

    def get_nb_invariant_polynomials(self, deg):
        return len(self.monomial_idx[deg])

    def get_invariant_polynomial(self, deg, idx):
        return self.monomial_idx[deg][idx], self.monomial_coeff[deg][idx]





