from spherical_harmonics.kernels import *
import numpy as np
import torch

def torch_fibonnacci_sphere_sampling(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    S2 = np.stack([x, y, z], axis=-1)
    return torch.from_numpy(S2).to(torch.float32)

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

def torch_eval_monoms_3D(x, d, axis=-1):
    m = monoms_3D(d)
    pows = np.zeros((3, len(m)), dtype=np.int32)
    for i in range(len(m)):
        for j in range(3):
            pows[j, i] = m[i][j]
    pows = torch.from_numpy(pows).to(torch.float32).to(x.device)
    n = len(list(x.shape))
    axis = axis % n
    shape = [1]*(n+1)
    shape[axis] = 3
    shape[-1] = len(m)
    pows = torch.reshape(pows, shape)
    x = x.unsqueeze(-1)
    y = torch.pow(x, pows)
    y = torch.prod(y, dim = axis, keepdims=False)
    return y

def np_monomial_basis_coeffs(polynomials, monoms_basis):
    n_ = len(monoms_basis)
    m_ = len(polynomials)
    M = np.zeros((m_, n_))
    for i in range(m_):
        for j in range(n_):
            M[i, j] = re(polynomials[i].coeff_monomial(monoms_basis[j]))
    return M

def torch_monomial_basis_coeffs(polynomials, monoms_basis, dtype=torch.float32):
    return torch.from_numpy(np_monomial_basis_coeffs(polynomials, monoms_basis)).to(dtype)

def torch_spherical_harmonics_(l, matrix_format=True):
    monoms = monoms_3D(l)
    sph_polys = []
    x, y, z = symbols("x y z")
    for m in range(2*l+1):
        sph_polys.append(real_spherical_harmonic(l, m-l, x, y, z, poly = True))
    coeffs = torch_monomial_basis_coeffs(sph_polys, monoms)
    if matrix_format:
        coeffs = torch.reshape(coeffs, (2*l+1, -1))
    return coeffs

def zernike_monoms(x, max_deg):
    m = int(max_deg / 2.)
    n2 = torch.sum(x * x, dim=-1, keepdims=True)
    n2 = n2.unsqueeze(-1)
    p = [torch.ones(n2.shape).type_as(x)]
    for m_ in range(m):
        p.append(p[-1] * n2)

    y = torch_spherical_harmonics(l_max=max_deg).compute(x)
    for l in y:
        y[l] = y[l].unsqueeze(-1)

    z = dict()
    for d in range(max_deg + 1):
        z[d] = []
    for l in y:
        l_ = int(l)
        for d in range(m + 1):
            d_ = 2 * d + l_
            if d_ <= max_deg:
                # print(p[d].shape)
                # print(y[l].shape)
                zd = (p[d] * y[l])
                z[l_].append(zd)
    for d in z:
        z[d] = torch.cat(z[d], dim=-1)
    return z


class torch_spherical_harmonics:
    def __init__(self, l_max=3, l_list=None):
        if l_list is None:
            self.l_list = range(l_max+1)
        else:
            self.l_list = l_list
        self.l_max = max(self.l_list)
        self.Y = dict()
        for l in self.l_list:
            self.Y[str(l)] = torch_spherical_harmonics_(l)

    def compute(self, x):
        Y = dict()
        for l in self.l_list:
            ml = torch_eval_monoms_3D(x, l)
            Y[str(l)] = torch.einsum('mk,...k->...m', self.Y[str(l)].type_as(ml), ml)
        return Y

def np_polyhedrons(poly):


    C0 = 3 * np.sqrt(2) / 4
    C1 = 9 * np.sqrt(2) / 8

    tetrakis_hexahedron = np.array([[0.0, 0.0, C1],
                                    [0.0, 0.0, -C1],
                                    [C1, 0.0, 0.0],
                                    [-C1, 0.0, 0.0],
                                    [0.0, C1, 0.0],
                                    [0.0, -C1, 0.0],
                                    [C0, C0, C0],
                                    [C0, C0, -C0],
                                    [C0, -C0, C0],
                                    [C0, -C0, -C0],
                                    [-C0, C0, C0],
                                    [-C0, C0, -C0],
                                    [-C0, -C0, C0],
                                    [-C0, -C0, -C0]], dtype=np.float32)

    C0 = (1 + np.sqrt(5)) / 4
    C1 = (3 + np.sqrt(5)) / 4

    regular_dodecahedron = np.array([[0.0, 0.5, C1], [0.0, 0.5, -C1], [0.0, -0.5, C1], [0.0, -0.5, -C1],
                      [C1, 0.0, 0.5], [C1, 0.0, -0.5], [-C1, 0.0, 0.5], [-C1, 0.0, -0.5],
                      [0.5, C1, 0.0], [0.5, -C1, 0.0], [-0.5, C1, 0.0], [-0.5, -C1, 0.0],
                      [C0, C0, C0], [C0, C0, -C0], [C0, -C0, C0], [C0, -C0, -C0],
                      [-C0, C0, C0], [-C0, C0, -C0], [-C0, -C0, C0], [-C0, -C0, -C0]], dtype=np.float32)

    C0 = 3 * (np.sqrt(5) - 1) / 4
    C1 = 9 * (9 + np.sqrt(5)) / 76
    C2 = 9 * (7 + 5 * np.sqrt(5)) / 76
    C3 = 3 * (1 + np.sqrt(5)) / 4

    pentakis_dodecahedron = np.array([[0.0, C0, C3], [0.0, C0, -C3], [0.0, -C0, C3], [0.0, -C0, -C3],
                      [C3, 0.0, C0], [C3, 0.0, -C0], [-C3, 0.0, C0], [-C3, 0.0, -C0],
                      [C0, C3, 0.0], [C0, -C3, 0.0], [-C0, C3, 0.0], [-C0, -C3, 0.0],
                      [C1, 0.0, C2], [C1, 0.0, -C2], [-C1, 0.0, C2], [-C1, 0.0, -C2],
                      [C2, C1, 0.0], [C2, -C1, 0.0], [-C2, C1, 0.0], [-C2, -C1, 0.0],
                      [0.0, C2, C1], [0.0, C2, -C1], [0.0, -C2, C1], [0.0, -C2, -C1],
                      [1.5, 1.5, 1.5], [1.5, 1.5, -1.5], [1.5, -1.5, 1.5], [1.5, -1.5, -1.5],
                      [-1.5, 1.5, 1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5], [-1.5, -1.5, -1.5]],
                     dtype=np.float)


    C0 = 3 * (15 + np.sqrt(5)) / 44
    C1 = (5 - np.sqrt(5)) / 2
    C2 = 3 * (5 + 4 * np.sqrt(5)) / 22
    C3 = 3 * (5 + np.sqrt(5)) / 10
    C4 = np.sqrt(5)
    C5 = (75 + 27 * np.sqrt(5)) / 44
    C6 = (15 + 9 * np.sqrt(5)) / 10
    C7 = (5 + np.sqrt(5)) / 2
    C8 = 3 * (5 + 4 * np.sqrt(5)) / 11

    disdyakis_triacontahedron = np.array([[0.0, 0.0, C8], [0.0, 0.0, -C8], [C8, 0.0, 0.0], [-C8, 0.0, 0.0],
                                          [0.0, C8, 0.0], [0.0, -C8, 0.0], [0.0, C1, C7], [0.0, C1, -C7],
                                          [0.0, -C1, C7], [0.0, -C1, -C7], [C7, 0.0, C1], [C7, 0.0, -C1],
                                          [-C7, 0.0, C1], [-C7, 0.0, -C1], [C1, C7, 0.0], [C1, -C7, 0.0],
                                          [-C1, C7, 0.0], [-C1, -C7, 0.0], [C3, 0.0, C6], [C3, 0.0, -C6],
                                          [-C3, 0.0, C6], [-C3, 0.0, -C6], [C6, C3, 0.0], [C6, -C3, 0.0],
                                          [-C6, C3, 0.0], [-C6, -C3, 0.0], [0.0, C6, C3], [0.0, C6, -C3],
                                          [0.0, -C6, C3], [0.0, -C6, -C3], [C0, C2, C5], [C0, C2, -C5],
                                          [C0, -C2, C5], [C0, -C2, -C5], [-C0, C2, C5], [-C0, C2, -C5],
                                          [-C0, -C2, C5], [-C0, -C2, -C5], [C5, C0, C2], [C5, C0, -C2],
                                          [C5, -C0, C2], [C5, -C0, -C2], [-C5, C0, C2], [-C5, C0, -C2],
                                          [-C5, -C0, C2], [-C5, -C0, -C2], [C2, C5, C0], [C2, C5, -C0],
                                          [C2, -C5, C0], [C2, -C5, -C0], [-C2, C5, C0], [-C2, C5, -C0],
                                          [-C2, -C5, C0], [-C2, -C5, -C0], [C4, C4, C4], [C4, C4, -C4],
                                          [C4, -C4, C4], [C4, -C4, -C4], [-C4, C4, C4], [-C4, C4, -C4],
                                          [-C4, -C4, C4], [-C4, -C4, -C4]], dtype=np.float32)

    P = {'tetrakis_hexahedron':tetrakis_hexahedron,
         'regular_dodecahedron':regular_dodecahedron,
         'pentakis_dodecahedron':pentakis_dodecahedron,
         'disdyakis_triacontahedron':disdyakis_triacontahedron}

    p = P[poly]
    c = np.mean(p, axis=0, keepdims=True)
    p = np.subtract(p, c)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    p = np.divide(p, n)
    return p

def torch_polyhedrons(poly):
    return torch.from_numpy(np_polyhedrons(poly)).to(torch.float32)

class SphericalHarmonicsEval:
    """
    Inverse Spherical Harmonics Transform layer
    """
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
            self.sph_fn = torch_spherical_harmonics(l_max=l_max, l_list=l_list)

        if isinstance(base, str):
            S2 = torch_polyhedrons(self.base)
        else:
            S2 = base


        y = self.sph_fn.compute(S2)
        self.types = y.keys()
        Y = []
        for l in self.types:
            Y.append(torch.reshape(y[l], (-1, 2*int(l)+1)))
        self.Y = torch.cat(Y, dim=-1)

    def compute(self, x):
        X = []
        for l in self.types:
            X.append(x[l])
        X = torch.cat(X, dim=-2)
        return torch.einsum('vm,...mc->...vc', self.Y.type_as(X), X)


class SphericalHarmonicsCoeffs:
    """
    Spherical Harmonics Transform layer
    """
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
            self.sph_fn = torch_spherical_harmonics(l_max=l_max, l_list=l_list)

        if isinstance(self.base, str):
            S2 = torch_polyhedrons(self.base)
        else:
            S2 = self.base

        y = self.sph_fn.compute(S2)

        self.types = list(y.keys())
        Y = []
        for l in self.types:
            Y.append(torch.reshape(y[l], (-1, 2*int(l)+1)))
        self.Y = torch.cat(Y, dim=-1)
        self.S2 = S2

    def compute(self, x):
        X = []
        c = torch.einsum('vm,...vc->...mc', self.Y.type_as(x), x) / (self.Y.shape[0] / (4*np.pi))
        c = torch.split(c, split_size_or_sections=self.split_size, dim=-2)

        C = dict()
        for i in range(len(self.types)):
            l = self.types[i]
            sl = list(x.shape)
            sl[-2] = 2*int(l)+1
            C[l] = torch.reshape(c[i], sl)
        return C

    def get_samples(self):
        return self.S2

if __name__=="__main__":

    device = "cuda:0"
    S2 = torch_fibonnacci_sphere_sampling(64).to(device)
    y = {}

    for i in range(4):
        y[str(i)] = (torch.ones((2, 16, 2*i + 1, 128)) * torch.arange(16).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).to(device)

    x = y.copy()
    # Inverse Spherical Harmonics Transform
    y = SphericalHarmonicsEval(l_max=3, base=S2).compute(y)
    print(y, y.shape)
    # Spherical Harmonics Transform
    y = SphericalHarmonicsCoeffs(l_max=3, base=S2).compute(y)
    for key in y:
        print(y[key], y[key].shape)
    pass