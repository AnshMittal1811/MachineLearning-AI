# ------ IMPORT LIBRARIES ------ #
import numpy as np
from scipy import linalg
from scipy.stats import random_correlation
import matplotlib.pyplot as plt
 
# define random data
rng = np.random.default_rng()
x = random_correlation.rvs((.5, .8, 1.2, 1.5), random_state=rng)
y = random_correlation.rvs((.2, .9, 1.1, 1.8), random_state=rng)

# generating 10 random values for each of the two variables
n = 50
p = 30
C_list = []

# computing the corrlation matrices
for i in range(n):
    # first corr matrix
    X1 = np.random.normal(-7, 1, p)
    Y1 = np.random.normal(5, 1, p)
    C1 = np.corrcoef(X1,Y1)
    # second corr matrix
    X2 = np.random.normal(3, 1.0, p)
    Y2 = np.random.normal(8, 1.0, p)
    C2 = np.corrcoef(X2,Y2)
    # add as list of pairs to compare distances
    C_list.append(np.array([C1, C2]))

# --------------------------------------------------
# ----- TEST CONDITIONS FOR SPD MATRICES -----------
# --------------------------------------------------

def is_spd(X, eps=1e-7):
    """Check matrix is symmetric & positive definite"""
    # X: Input n x n matrix 
    # Check X = X^T and min eigenvalue > 0
    if np.any(np.abs(X - X.T) > eps):
        raise ValueError('Error: input matrix must be symmetric')
    eigvals = linalg.eigvals(X)
    if min(eigvals) <= 0:
        raise ValueError('Error: input matrix has non-positive eigenvalue')
    return True


# --------------------------------------------------------
# ----- HELPER DISTANCE/GEOMETRY FUNCTIONS on SPD(M) -----
# --------------------------------------------------------

# source: https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/geodesic.py

def distance_euclid(A, B):
    r"""Euclidean distance between two covariance matrices A and B.
    The Euclidean distance is defined by the Froebenius norm between the two
    matrices.
    .. math::
        d = \Vert \mathbf{A} - \mathbf{B} \Vert_F
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Eclidean distance between A and B
    """
    return np.linalg.norm(A - B, ord='fro')


def distance_logeuclid(A, B):
    r"""Log Euclidean distance between two covariance matrices A and B.
    .. math::
        d = \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Eclidean distance between A and B
    """
    return distance_euclid(linalg.logm(A), linalg.logm(B))


def distance_riemann(A, B):
    r"""Riemannian distance between two covariance matrices A and B.
    .. math::
        d = {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}
    where :math:`\lambda_i` are the joint eigenvalues of A and B
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Riemannian distance between A and B
    """
    return np.sqrt((np.log(linalg.eigvalsh(A, B))**2).sum())

def distance_logdet(A, B):
    r"""Log-det distance between two covariance matrices A and B.
    .. math::
        d = \sqrt{\log(\det(\frac{\mathbf{A}+\mathbf{B}}{2})) - \frac{1}{2} \log(\det(\mathbf{A}) \det(\mathbf{B}))}
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Euclid distance between A and B
    """  # noqa
    return np.sqrt(np.log(np.linalg.det(
        (A + B) / 2.0)) - 0.5 *
        np.log(np.linalg.det(A)*np.linalg.det(B)))


def distance_wasserstein(A, B):
    r"""Wasserstein distance between two covariances matrices.
    .. math::
        d = \left( {tr(A + B - 2(A^{1/2}BA^{1/2})^{1/2})} \right)^{1/2}
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Wasserstein distance between A and B
    """
    B12 = sqrtm(B)
    C = sqrtm(np.dot(np.dot(B12, A), B12))
    return np.sqrt(np.trace(A + B - 2*C))


# ----------------
# -- TESTING -----
# ----------------
print(np.all([is_spd(c[0]) and is_spd(c[1]) for c in C_list]))
riemann_dists = np.array([distance_riemann(c[0], c[1]) for c in C_list])
logeuclid_dists = np.array([distance_logeuclid(c[0], c[1]) for c in C_list])
euclid_dists = np.array([distance_euclid(c[0], c[1]) for c in C_list])
plt.scatter(euclid_dists, riemann_dists)
plt.hist(riemann_dists)
plt.hist(euclid_dists)


# -------------------------------------------------------------
# ----- FINISH GOING THROUGH PAPER TO IMPLEMENT PRECISELY -----
# -------------------------------------------------------------

def spd_dist(X, Y, metric='intrinsic'):
    """Calculate geodesic distance for X,Y in SPD Manifold"""
    # X: Input n x n matrix
    # Y: Input n x n matrix
    # Intrinsic metric: Affine-invariant Riemannian Metric (AIRM)
    # Extrinsic metric: log-Euclidean Riemannian Metric (LERM)
    if metric == 'intrinsic':
        M = np.matmul(linalg.inv(X), Y)
        dist = np.sqrt(np.linalg.norm(linalg.logm(M)))
        return dist
    elif metric == 'extrinsic':
        M = linalg.logm(X) - linalg.logm(Y)
        dist = np.sqrt(np.linalg.norm(M))
        return dist
    else:
        raise ValueError('Error: must specify intrinsic or extrinsic metric')
        
#spd_dists = np.array([spd_dist(c[0], c[1]) for c in C_list])
spd_dists = np.array([spd_dist(c[0], c[1], metric='extrinsic') for c in C_list])
        
def exp_map(X, V):
    """Exponential mapping from tangent space at X to SPD Manifold"""
    # X: n x n matrix in SPD Manfiold (M)
    # V: tangent "vector" (really a symmetric matrix) within Tx(M)
    # Output: a point Y in M (following shortest geodesic curve along M in direction v)
    # NOTE: tangent "vectors" in Tx(M) are n x n symmetric matrics

#   -- remember matlab docs on using eigenvals to calculate inverse square roots, 
#   -- cholesky/decomposition for SPD
#   -- Expai,X(U) = X1/2 exp(X−1U)X1/2 (new formula from ADHD pdf)
    
    
    
    
# ---------------------------------------------------
# -- TESTING FORMULA TO USE FOR EXP/LOG MAPPING -----
# ---------------------------------------------------

def sqrtm(Ci):
    r"""Return the matrix square root of a covariance matrix defined by :
    .. math::
        \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \right)^{1/2} \mathbf{V}^\top
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the coavriance matrix
    :returns: the matrix square root
    """  # noqa
    return _matrix_operator(Ci, np.sqrt)


def logm(Ci):
    r"""Return the matrix logarithm of a covariance matrix defined by :
    .. math::
        \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^\top
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the covariance matrix
    :returns: the matrix logarithm
    """
    return _matrix_operator(Ci, np.log)


def expm(Ci):
    r"""Return the matrix exponential of a covariance matrix defined by :
    .. math::
        \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^\top
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the coavriance matrix
    :returns: the matrix exponential
    """
    return _matrix_operator(Ci, np.exp)


def invsqrtm(Ci):
    r"""Return the inverse matrix square root of a covariance matrix defined by :
    .. math::
        \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \right)^{-1/2} \mathbf{V}^\top
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root
    """  # noqa
    def isqrt(x): return 1. / np.sqrt(x)
    return _matrix_operator(Ci, isqrt)


# WE USE: A = Q*D*Q^T (if A real symmetric)
test_X = x # simple test case to learn
operator = np.sqrt # define for any operation we need to use in exp/log map

# We get eigenvals/vecs from original SPD Matrix
eigvals, eigvects = linalg.eigh(test_X, check_finite=False)
print(eigvals)
print(eigvects)
# We get eigenvals to form diagonal matrix D 
eigvals = np.diag(operator(eigvals))
print(eigvals)
# We multiple to get eigen decomposition: eigenvectors are columns of Q
np.dot(np.dot(eigvects, eigvals), eigvects.T)

linalg.sqrtm(test_X)

# NOTE: We can apply this to square root, log, exp, and inverse matrix operations
# A^-1 = Q*D^-1*QT





        
    
    
    
    
        

        