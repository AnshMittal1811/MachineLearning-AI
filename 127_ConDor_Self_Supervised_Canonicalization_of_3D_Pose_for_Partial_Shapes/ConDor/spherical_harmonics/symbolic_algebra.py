from sympy import *
from sympy.polys.monomials import itermonomials, monomial_count
from sympy.polys.orderings import monomial_key
x, y, z = symbols("x y z")
a, b, c = symbols("a b c")


from scipy.special import gamma
import numpy as np

def binom(n, k):
    if k == 0.:
        return 1.
    return gamma(n + 1) / (gamma(n-k+1)*gamma(k+1))

def zernike_polynomial_radial(l,n,D, x, y, z):
    if (l > n):
        return 0
    if ((n-l) % 2 != 0):
        return 0
    rho2 = x**2 + y**2 + z**2
    # rho = sqrt(rho2)
    R = 0

    for s in range(int((n-l) / 2) + 1):
        R += (-1)**s * binom((n-l)/2, s)*binom(s-1+(n+l+D)/2., (n-l)/2)*rho2**(s)
    R *= (-1)**((n-l)/2)*sqrt(2*n+D)
    """
    for k in range(int((n-l) / 2) + 1):
        R += (-1)**k * binom(n-k, k)*binom(n-2*k, (n-l)/2 - k)*rho**(n-2*k)
    """
    return R

# print(zernike_polynomial_radial(0,0,3))

"""
l = 0, n = 0, n = 2, n = 4
l = 1, n = 1, n = 3, n = 5
l = 2, n = 2, n = 4, n = 6
l = 3, n = 3, n = 5, n = 7
"""

"""
Spherical harmonics
"""



def spherical_harmonics(x, y, z):
    Y00 = 0.5*sqrt(1/np.pi)

    Y1_1 = (np.sqrt(3. / np.pi) / 2.) * y
    Y10 = (np.sqrt(3. / np.pi) / 2.) * z
    Y11 = (np.sqrt(3. / np.pi) / 2.) * x

    Y2_2 = (np.sqrt(15. / np.pi) / 2.) * x*y
    Y2_1 = (np.sqrt(15. / np.pi) / 2.) * y*z
    Y20 = (np.sqrt(5. / np.pi) / 4.) * (2. * z**2 - x**2 - y**2)
    Y21 = (np.sqrt(15. / np.pi) / 2.) * z*x
    Y22 = (np.sqrt(15. / np.pi) / 4.) * (x**2 - y**2)

    Y3_3 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (3. * x**2 - y**2)*y
    Y3_2 = (np.sqrt(105. / np.pi) / 2.) * x*y*z
    Y3_1 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * z**2 - x**2 - y**2)*y
    Y30 = (np.sqrt(7. / np.pi) / 4.) * (2. * z**2 - 3. * x**2 - 3. * y**2)*z
    Y31 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * z**2 - x**2 - y**2)*x
    Y32 = (np.sqrt(105. / np.pi) / 4.) * (x**2 - y**2)*z
    Y33 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (x**2 - 3. * y**2)*x

    Y = dict([])
    Y['Y_{0,0}'] = Y00

    Y['Y_{1,-1}'] = Y1_1
    Y['Y_{1,0}'] = Y10
    Y['Y_{1,1}'] = Y11


    Y['Y_{2,-2}'] = Y2_2
    Y['Y_{2,-1}'] = Y2_1
    Y['Y_{2,0}'] = Y20
    Y['Y_{2,1}'] = Y21
    Y['Y_{2,2}'] = Y22

    Y['Y_{3,-3}'] = Y3_3
    Y['Y_{3,-2}'] = Y3_2
    Y['Y_{3,-1}'] = Y3_1
    Y['Y_{3,0}'] = Y30
    Y['Y_{3,1}'] = Y31
    Y['Y_{3,2}'] = Y32
    Y['Y_{3,3}'] = Y33

    return Y

def zerinke_polynomial3D(n, l, m, x, y, z):
    Y = spherical_harmonics(x, y, z)
    Y_key = 'Y_{' + str(l) + ',' + str(m) + '}'
    Z = expand(zernike_polynomial_radial(l, n, 3, x, y, z) * Y[Y_key])
    return Z

def zernike3D(x, y, z):
    """
    l = 0, n = 0, n = 2, n = 4
    l = 1, n = 1, n = 3, n = 5
    l = 2, n = 2, n = 4, n = 6
    l = 3, n = 3, n = 5, n = 7
    """
    Z = dict()
    C = [[0, 0], [0, 2], [1, 1], [1, 3], [2, 2], [3, 3]]

    # C = [[0, 0], [0, 2], [0, 4], [1, 1], [1, 3], [2, 2], [2, 4], [3, 3]]

    for c in C:
        l = c[0]
        n = c[1]
        for m_ in range(2*l+1):
            m = m_ - l
            Z_key = 'Z_{' + str(n) + ',' + str(l) + ',' + str(m) + '}'
            Z[Z_key] = zerinke_polynomial3D(n, l, m, x, y, z)
    return Z

def zerinke_polynomials3D(nr, l_max, x, y, z):
    Z = dict([])
    Y = spherical_harmonics(x, y, z)
    for l in range(l_max+1):
        n = l
        for j in range(nr):
            for m_ in range(2*l+1):
                m = m_ - l
                Y_key = 'Y_{' + str(l) + ',' + str(m) + '}'
                Z_key = 'Z_{' + str(n) + ',' + str(l) + ',' + str(m) + '}'
                Z[Z_key] = expand(zernike_polynomial_radial(l, n, 3, x, y, z) * Y[Y_key])
            n += 2
    return Z


#Z = zerinke_polynomials3D(2, 3, x-a, y-b, z-c)
Z = zernike3D(x-a, y-b, z-c)

"""
print(Z['Z_{3,3,0}'])
P = poly(Z['Z_{3,3,0}'], x, y, z)
print(P.monoms())
Q = 3*x + z - 2*y
print(Q.coeff(x))
"""
"""
M = []
for key in Z:
    P = poly(Z[key], x, y, z)
    M += P.monoms()

print(len(M))
M = set(M)
print(M)
print(len(M))
"""

ZZ = dict()
monoms_basis = []


for key in Z:
    P = poly(Z[key], x, y, z)
    M = P.monoms()
    monoms_basis += M
    C = [Poly(P.coeff_monomial(m), a, b, c) for m in M]
    CM = [c.monoms() for c in C]
    CCM = [[[CM[i][j], C[i].coeff_monomial(CM[i][j])] for j in range(len(CM[i]))] for i in range(len(M))]
    ZZ[key] = [[M[i], CCM[i]] for i in range(len(M))]

monoms_basis = list(set(monoms_basis))
monoms_basis = sorted(monoms_basis)

print(monoms_basis)

idx = dict()
for i in range(len(monoms_basis)):
    idx[monoms_basis[i]] = i
N = len(monoms_basis)
A = dict()
for key in ZZ:
    a = np.zeros((N, N))
    print(ZZ[key][0][0])
    for i in range(len(ZZ[key])):
        for j in range(len(ZZ[key][i][1])):
            a[idx[ZZ[key][i][0]], idx[ZZ[key][i][1][j][0]]] += ZZ[key][i][1][j][1]
    A[key] = a

# for m in ZZ[]

print(A)

"""
print(P)
print(P.monoms())
print([prod(x**k for x, k in zip(P.gens, mon)) for mon in P.monoms()])


print(sorted(itermonomials([x, y, z], 3), key=monomial_key('grlex', [z, y, x])))
print(monomial_count(3, 3))
"""

