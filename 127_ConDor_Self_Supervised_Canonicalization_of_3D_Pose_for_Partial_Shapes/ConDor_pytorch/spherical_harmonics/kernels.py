from sympy import *
import numpy as np
import torch, h5py

from utils.group_points import gather_idx, GroupPoints
from spherical_harmonics.clebsch_gordan_decomposition import torch_clebsch_gordan_decomposition

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
def real_spherical_harmonic(l, m, x, y, z, poly = False):
    K = np.sqrt((2*l+1)/(2*np.pi))
    r2 = x**2 + y**2 + z**2
    if m > 0:
        Ylm = K * associated_legendre_polynomial(l, m, z, r2) * A(m, x, y)
    elif m < 0:
        Ylm = K * associated_legendre_polynomial(l, -m, z, r2) * B(-m, x, y)
    else:
        K = np.sqrt((2 * l + 1) / (4 * np.pi))
        Ylm = K * associated_legendre_polynomial(l, 0, z, r2)
    if poly:
        Ylm = Poly(simplify(expand(Ylm)), x, y, z)
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

def torch_monomial_basis_3D_idx(d):
    m = monomial_basis_3D(d)
    idx = np.zeros((len(m), 3), dtype=np.int32)
    for i in range(len(m)):
        for j in range(3):
            idx[i, j] = m[i][j]
    return torch.from_numpy(idx).to(torch.int64)

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


class ShGaussianKernelConv(torch.nn.Module):
    def __init__(self, l_max, l_max_out=None, transpose=False, num_source_points=None):
        super(ShGaussianKernelConv, self).__init__()
        self.l_max = l_max
        self.split_size = []
        for l in range(l_max + 1):
            self.split_size.append(2 * l + 1)
        # self.output_type = output_type
        self.l_max_out = l_max_out
        # self.transpose = transpose
        self.num_source_points = num_source_points
        
        
        # Clebsh Gordon coeffs decompose a tensor into expansion of eigenstates of angular momentum
        # Youtube video: https://www.youtube.com/watch?v=UPyf9ntr-B8
        # Couples two different type l features with each other 
        self.Q = torch_clebsch_gordan_decomposition(l_max=max(l_max_out, l_max),
                                                 sparse=False,
                                                 output_type='dict',
                                                 l_max_out=l_max_out)

    def forward(self, x):
        assert (isinstance(x, dict))

        signal = []
        features_type = []
        channels_split_size = []
        for l in x:
            if l.isnumeric():
                features_type.append(int(l))
                channels_split_size.append(x[l].shape[-2] * x[l].shape[-1])
                signal.append(torch.reshape(x[l], (x[l].shape[0], x[l].shape[1], -1)))


        signal = torch.cat(signal, dim=-1)
        batch_size = signal.shape[0]
        patch_size = x["kernels"].shape[2]
        num_shells = x["kernels"].shape[-1]

        # Changed and removed transpose here
        if "patches idx" in x:
            # print(signal.shape, "signal shape", x["patches idx"].shape)
            signal = gather_idx(signal, x["patches idx"])
            # print(signal.shape)

        num_points_target = signal.shape[1]
        kernels = torch.reshape(x["kernels"], (batch_size, num_points_target, patch_size, -1))
        y = torch.einsum('bvpy,bvpc->bvyc', kernels, signal)



        # split y
        # print(channels_split_size, y.shape)
        y_ = torch.split(y, split_size_or_sections=channels_split_size, dim=-1)
        y = {str(j): [] for j in range(self.l_max_out + 1)}
        y_cg = []
        for i in range(len(channels_split_size)):
            l = features_type[i]
            yi = torch.reshape(y_[i], (batch_size, num_points_target, -1, num_shells, 2 * l + 1, x[str(l)].shape[-1]))
            yi = yi.permute(0, 1, 2, 4, 3, 5)
            yi = torch.reshape(yi, (batch_size, num_points_target, -1, 2 * l + 1, num_shells*x[str(l)].shape[-1]))
            yi = torch.split(yi, split_size_or_sections=self.split_size, dim=2)
            for j in range(len(self.split_size)):

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
            y[J] = torch.cat(y[J], dim=-1)
        return y



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
    return M # 2l + 1 , n_

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




"""
computes the 3D spherical harmonics basis up to degree l_max
"""
def torch_spherical_harmonics_basis(l_max, concat=False):
    monoms_basis = monomial_basis_3D(l_max)
    Y = []
    for l in range(l_max+1):
        Yl = spherical_harmonics_3D_monomial_basis(l, monoms_basis)
        Y.append(torch.from_numpy(Yl).to(torch.float32))
    if concat:
        Y = torch.cat(Y, dim=0)
    return Y

def np_zernike_kernel(d, n, l):
    monoms_basis = monomial_basis_3D(d)
    assert (n >= l and (n - l) % 2 == 0)
    return zernike_kernel_3D_monomial_basis(n, l, monoms_basis)

def torch_eval_monom_basis(x, d, idx=None):
    """
    evaluate monomial basis up to degree d
    x - B, N, K, 3

    y - B, N, K, 20
    """

    batch_size = x.shape[0]
    num_points = x.shape[1]

    if idx is None:
        idx = torch_monomial_basis_3D_idx(d)
    y = []
    for i in range(3):
        pows = torch.reshape(torch.arange(d+1), (1, 1, d+1)).to(torch.float32) # 1, 1, num_pow
        yi = torch.pow(x[..., i].unsqueeze(-1), pows.type_as(x)) # B, N, K, num_pow
        y.append(yi[..., idx[..., i]])        
    y = torch.stack(y, dim=-1)
    y = torch.prod(y, dim=-1, keepdims=False)
    return y





class SphericalHarmonicsGaussianKernels(torch.nn.Module):

    """
        Computes steerable kernel for point clouds. Eqn. 13 and Eqn. 4 of paper
        https://openaccess.thecvf.com/content/CVPR2021/papers/Poulenard_A_Functional_Approach_to_Rotation_Equivariant_Non-Linearities_for_Tensor_Field_CVPR_2021_paper.pdf
    
    """
    def __init__(self, l_max, gaussian_scale, num_shells, transpose=False, bound=True):
        super(SphericalHarmonicsGaussianKernels, self).__init__()
        self.l_max = l_max
        self.monoms_idx = torch_monomial_basis_3D_idx(l_max)
        self.gaussian_scale = gaussian_scale
        self.num_shells = num_shells
        self.transpose = True
        self.Y = torch_spherical_harmonics_basis(l_max, concat=True)
        self.split_size = []
        self.sh_idx = []
        self.bound = bound
        for l in range(l_max + 1):
            self.split_size.append(2*l+1)
            self.sh_idx += [l]*(2*l+1)
        self.sh_idx = torch.from_numpy(np.array(self.sh_idx)).to(torch.int64)



    def forward(self, x):
        
        if "patches dist" in x:
            patches_dist = x["patches dist"].unsqueeze(-1) # B, N, K, 1
        else:
            patches_dist = torch.linalg.norm(x["patches"], dim=-1, keepdims=True)

        normalized_patches = x["patches"] / torch.maximum(patches_dist, torch.tensor(0.000001).type_as(x["patches"])) # Normalize by max dist to make it [0, 1] B, N, K, 3

        if self.transpose:
            normalized_patches = -normalized_patches
        
        # Obtain spherical harmonics for each patch
        monoms_patches = torch_eval_monom_basis(normalized_patches, self.l_max, idx=self.monoms_idx) # B, N, K, 20 (sum 2*l+1 less than or equal)
        # print(monoms_patches.shape, self.Y.shape)

        # Y^l_m(x_j - y_i)
        sh_patches = torch.einsum('ij,bvpj->bvpi', self.Y.type_as(monoms_patches), monoms_patches) # B, N, K, 16 / 9

        # shells_rad = r / (n_r - 1) = rj
        shells_rad = torch.arange(self.num_shells).type_as(monoms_patches) / (self.num_shells-1)
        shells_rad = torch.reshape(shells_rad, (1, 1, 1, -1)) # 1, 1, 1, num_shells
        
        # x^2 - rj
        shells = patches_dist - shells_rad # B, N, K, 1 - 1, 1, 1, num_shells = B, N, K, num_shells

        # self.gaussian_scale = -ln(2)*d**2
        # shells = exp(-ln(2) * d**2 * (x_2 - r_j)**2)
        shells = torch.exp(-self.gaussian_scale*(shells * shells))

        # denominator is shells_sum
        shells_sum = torch.sum(shells, dim=-1, keepdims=True)
        
        # compute kernel 
        # shells = exp(-ln(2) * d**2 * (x_2 - r_j)**2) / sum_{k=0}^{d-1} exp(-ln(2) * d**2 * (x_2 - r_k)**2)
        shells = (shells / torch.maximum(shells_sum, torch.tensor(0.000001).type_as(shells))) # B, N, K, num_shells

        # Steerable kernel calculated
        shells = shells.unsqueeze(-2) # B, N, K, 1, num_shells
        if self.bound:
            shells = torch.where(patches_dist.unsqueeze(-1) <= torch.tensor(1.).type_as(shells), shells, torch.tensor(0.).type_as(shells))

        sh_patches = sh_patches.unsqueeze(-1)

        # Shells * spherical harmonics of points
        sh_patches = shells * sh_patches


        # L2 norm
        # Calculate c_{irl} term (Eqn 14)
        # \sum_{j \sim i} (x_j^{k} - x_i^{k + 1})^{2}
        l2_norm = torch.sum((sh_patches * sh_patches), dim=2, keepdims=True)
        l2_norm = torch.split(l2_norm, split_size_or_sections=self.split_size, dim=-2)
        Y = []
        for l in range(len(l2_norm)):
            ml = torch.sum(l2_norm[l], dim=-2, keepdims=True)
            ml = torch.sqrt(ml + 1e-7)
            Y.append(ml)
        l2_norm = torch.cat(Y, dim=-2)
        l2_norm = torch.mean(l2_norm, dim=1, keepdims=True)
        l2_norm = torch.maximum(l2_norm, torch.tensor(1e-8).type_as(l2_norm))
        # print(l2_norm.shape)
        l2_norm = l2_norm[..., self.sh_idx, :]

        # divide by c_{irl}
        sh_patches = (sh_patches / (l2_norm + 1e-6))

        return sh_patches






if __name__=="__main__":

    filename = "/home/rahul/research/data/sapien_processed/train_refrigerator.h5"
    f = h5py.File(filename, "r")
    x = torch.from_numpy(f["data"][:2]).cuda()
    x2 = torch.from_numpy(f["data"][2:4]).cuda()

    # print(x.shape)
    # x = (torch.ones((1, 4, 3)) * torch.arange(4).unsqueeze(0).unsqueeze(-1)).cuda()
    # y = torch_eval_monom_basis(x, 3)
    # print(x)
    # print(y, y.shape)
    gi = GroupPoints(0.2, 32)
    out = gi({"source points": x, "target points": x2})
    
    sph_kernels = SphericalHarmonicsGaussianKernels(l_max = 3, gaussian_scale = 0.1, num_shells = 3, bound = True).cuda()
    # x = (torch.ones((1, 100, 32, 3)) * torch.arange(3).unsqueeze(0).unsqueeze(1).unsqueeze(2)).cuda()
    # print(x.shape)
    patches = sph_kernels({"patches": out["patches source"], "patches dist": out["patches dist source"]})
    conv_layer = ShGaussianKernelConv(l_max=3, l_max_out=3).cuda()
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
        y['1'] = x2.unsqueeze(-1)

    y = conv_layer(y)
    # print(y, y.shape)
    for key in y:
        print(y[key], " ", key, " ", y[key].shape)
    
    
    # print(patches, patches.shape)