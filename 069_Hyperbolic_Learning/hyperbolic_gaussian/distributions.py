# import libraries
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sys
import os

# import modules within repository
sys.path.append('C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\utils') # path to utils folder
from utils import *

#------------------------------------------------------------
#----- Wrapped Normal Distribution in Hyperboloid Model -----
#------------------------------------------------------------

# first get sample from standard multivariate gaussian 
def init_sample(dim=2, variance=None):
    """Sample v from normal distribution in R^n+1 with N(0, sigma)"""
    mean = np.zeros((dim))
    if variance is None:
        variance = np.eye(dim)
    v = np.random.multivariate_normal(mean, variance)
    tangent_0 = np.insert(v, 0, 0)
    return tangent_0

# define alternate minkowski/hyperboloid bilinear form
def lorentz_product(u, v):
    """Compute lorentz product with alternate minkowski/hyperboloid bilinear form"""
    return -minkowski_dot(u, v)

def lorentz_norm(u, eps=1e-5):
    """Compute norm in hyperboloid using lorentz product"""
    return np.sqrt(np.max([lorentz_product(u,u), eps]))

def parallel_transport(transport_vec, target_vec, base_vec, eps=1e-5):
    """Mapping between tangent spaces, transports vector along geodesic from v to u""" 
    alpha = -lorentz_product(base_vec, target_vec)
    frac = lorentz_product(target_vec - alpha*base_vec, transport_vec) / (alpha+1+eps)
    return transport_vec + frac*(base_vec + target_vec)

def exponential_map(u, mu):
    """Given v in tangent space of u, we project v onto the hyperboloid surface""" 
    first = np.cosh(lorentz_norm(u)) * mu 
    last = np.sinh(lorentz_norm(u)) * (u / lorentz_norm(u))
    return first + last

def logarithm_map(z, mu, eps=1e-5):
    """Given z in hyperboloid, we project z onto the tangent space at mu""" 
    alpha = -lorentz_product(mu, z)
    numer = np.arccosh(alpha) * (z - alpha*mu) 
    denom = np.sqrt(max(alpha**2 - 1, eps))
    return numer / denom

def hyperbolic_sampling(n_samples, mean, sigma, dim=2, poincare=False):
    """Generate n_samples from the wrapped normal distribution in hyperbolic space"""
    data = []
    mu_0 = np.insert(np.zeros((dim)), 0, 1) 
    for i in range(n_samples):
        init_v = init_sample(dim=dim, variance=sigma)
        tangent_u = parallel_transport(base_vec=mu_0, target_vec=mean, transport_vec=init_v)
        data.append(exponential_map(tangent_u, mean))
    data = np.array(data)
    if poincare:
        return hyperboloid_pts_to_poincare(data, metric='minkowski')
    else:
        return data

def log_pdf(z, mu, sigma):
    """Given sample z and parameters mu, sigma calculate log of p.d.f(z)""" 
    n = len(z) - 1
    mu_0 = np.insert(np.zeros((n)), 0, 1)
    u = logarithm_map(z, mu)
    v = parallel_transport(transport_vec=u, target_vec=mu_0, base_vec=mu)
    r = lorentz_norm(u)
    det_proj = (np.sinh(r) / r)**(n-1)
    pv = multivariate_normal.pdf(v[1:], mean=np.zeros((n)), cov=sigma)
    return np.log10(pv) - np.log10(det_proj)