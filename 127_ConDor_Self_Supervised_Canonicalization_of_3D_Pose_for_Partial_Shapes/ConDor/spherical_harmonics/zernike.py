from sympy import *
import numpy as np

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
    R *= ((-1)**((n-l)/2))*np.sqrt(2*n+D)
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

"""
computes the coefficients of the Zernike polynomials in the monomial basis
"""
def zernike_kernel_3D_monomial_basis(n, l, monoms_basis):
    print("Z_{" + str(n) + ',' + str(l) + "}")
    print("")
    x, y, z = symbols("x y z")
    n_ = len(monoms_basis)
    M = np.zeros((2*l+1, n_))
    for m in range(2*l+1):
        Z = zernike_kernel_3D(n, l, m-l, x, y, z)
        Z = expand(Z)
        print("Z_{" + str(n) + ',' + str(l) + "," + str(m-l) + "}")
        Z = poly(Z, x, y, z)
        print(Z)
        for i in range(n_):
            M[m, i] = N(Z.coeff_monomial(monoms_basis[i]))
    print("")
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

"""
computes the 3D zernike basis up to degree d
"""
def np_zernike_kernel_basis(d):
    monoms_basis = monomial_basis_3D(d)
    Z = []
    for l in range(d+1):
        Zl = []
        for n in range(l, d+1):
            if (n - l) % 2 == 0 and d >= n:
                Zl.append(zernike_kernel_3D_monomial_basis(n, l, monoms_basis))
        Z.append(np.stack(Zl, axis=0))
    return Z

def np_zernike_kernel(d, n, l):
    monoms_basis = monomial_basis_3D(d)
    assert (n >= l and (n - l) % 2 == 0)
    return zernike_kernel_3D_monomial_basis(n, l, monoms_basis)

np_zernike_kernel_basis(3)

