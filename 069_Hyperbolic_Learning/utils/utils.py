import numpy as np

def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

#-------------------------
#----- Poincaré Disk -----
#-------------------------

# NOTE: POSSIBLE ISSUE WITH DIFFERENT WAYS TO SPECIFY MINKOWSKI DOT PRODUCT
# arbritray sign gives different signatures (+, +, +, -), (+, -, -, -)
    
# distance in poincare disk
def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)

# compute symmetric poincare distance matrix
def poincare_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = poincare_dist(embedding[i], embedding[j])
    return dist_matrix

# convert array from poincare disk to hyperboloid
def poincare_pts_to_hyperboloid(Y, eps=1e-6, metric='lorentz'):
    mink_pts = np.zeros((Y.shape[0], Y.shape[1]+1))
    r = norm(Y, axis=1)
    if metric == 'minkowski':
        mink_pts[:, 0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        mink_pts[:, 1] = 2/(1 - r**2 + eps) * Y[:, 0]
        mink_pts[:, 2] = 2/(1 - r**2 + eps) * Y[:, 1]
    else:
        mink_pts[:, 0] = 2/(1 - r**2 + eps) * Y[:, 0]
        mink_pts[:, 1] = 2/(1 - r**2 + eps) * Y[:, 1]
        mink_pts[:, 2] = 2/(1 - r**2 + eps) * (1 + r**2)/2
    return mink_pts

# convert single point to hyperboloid
def poincare_pt_to_hyperboloid(y, eps=1e-6, metric='lorentz'):
    mink_pt = np.zeros((3, ))
    r = norm(y)
    if metric == 'minkowski':
        mink_pt[0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        mink_pt[1] = 2/(1 - r**2 + eps) * y[0]
        mink_pt[2] = 2/(1 - r**2 + eps) * y[1]
    else:
        mink_pt[0] = 2/(1 - r**2 + eps) * y[0]
        mink_pt[1] = 2/(1 - r**2 + eps) * y[1]
        mink_pt[2] = 2/(1 - r**2 + eps) * (1 + r**2)/2
    return mink_pt

#------------------------------
#----- Hyperboloid Model ------
#------------------------------

# NOTE: POSSIBLE ISSUE WITH DIFFERENT WAYS TO SPECIFY MINKOWSKI DOT PRODUCT
# arbritray sign gives different signatures (+, +, +, -), (+, -, -, -)

# define hyperboloid bilinear form
def hyperboloid_dot(u, v):
    return np.dot(u[:-1], v[:-1]) - u[-1]*v[-1]

# define alternate minkowski/hyperboloid bilinear form
def minkowski_dot(u, v):
    return u[0]*v[0] - np.dot(u[1:], v[1:]) 

# hyperboloid distance function
def hyperboloid_dist(u, v, eps=1e-6, metric='lorentz'):
    if metric == 'minkowski':
        dist = np.arccosh(-1*minkowski_dot(u, v))
    else:
        dist = np.arccosh(-1*hyperboloid_dot(u, v))
    if np.isnan(dist):
        #print('Hyperboloid dist returned nan value')
        return eps
    else:
        return dist

# compute symmetric hyperboloid distance matrix
def hyperboloid_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = hyperboloid_dist(embedding[i], embedding[j])
    return dist_matrix

# convert array to poincare disk
def hyperboloid_pts_to_poincare(X, eps=1e-6, metric='lorentz'):
    poincare_pts = np.zeros((X.shape[0], X.shape[1]-1))
    if metric == 'minkowski':
        poincare_pts[:, 0] = X[:, 1] / ((X[:, 0]+1) + eps)
        poincare_pts[:, 1] = X[:, 2] / ((X[:, 0]+1) + eps)
    else:
        poincare_pts[:, 0] = X[:, 0] / ((X[:, 2]+1) + eps)
        poincare_pts[:, 1] = X[:, 1] / ((X[:, 2]+1) + eps)
    return poincare_pts

# project within disk
def proj(theta,eps=0.1):
    if norm(theta) >= 1:
        theta = theta/norm(theta) - eps
    return theta

# convert single point to poincare
def hyperboloid_pt_to_poincare(x, eps=1e-6, metric='lorentz'):
    poincare_pt = np.zeros((2, ))
    if metric == 'minkowski':
        poincare_pt[0] = x[1] / ((x[0]+1) + eps)
        poincare_pt[1] = x[2] / ((x[0]+1) + eps)
    else:
        poincare_pt[0] = x[0] / ((x[2]+1) + eps)
        poincare_pt[1] = x[1] / ((x[2]+1) + eps)
    return proj(poincare_pt)
    
# helper function to generate samples
def generate_data(n, radius=0.7, hyperboloid=False):
    theta = np.random.uniform(0, 2*np.pi, n)
    u = np.random.uniform(0, radius, n)
    r = np.sqrt(u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    init_data = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    if hyperboloid:
        return poincare_pts_to_hyperboloid(init_data)
    else:
        return init_data