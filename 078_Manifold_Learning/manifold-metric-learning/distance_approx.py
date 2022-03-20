import numpy as np
from scipy import integrate
import sys
my_path = 'C:\\Users\\dreww\\Desktop\\manifold-metric-learn' # path to utils folder
sys.path.append(my_path)
from example_manifolds import *

def approximate_distance(a, b, integrand, L=None, n=7, m=50, tol=1e-6):
    """
    Approximate manifold distance between the diffeomorphic mappings F(a)=x and F(b)=y 
    Parameters
    ----------
    a : F^-1 (x), point in Euclidean base space B, initial point in linear sequence a0, ..., a_n+1
    b : F^-1 (y), point in Euclidean base space B, endpoint of linear sequence a0, ..., a_n+1
    L : optional linear transformation matrix L applied to B, if none use identity matrix
    n : number of intermediate points between a,b in sequence a0, a1, ..., a_n+1
    m : number of bi sampled from base space at each intermediate point
    """
    base_dim = len(a)
    if L == None:
        L = np.eye(base_dim)
    if not (len(a) == len(b) == L.shape[0]):
        print('Input dimensions do not match!')
        return
    
    # create list with sequence a0, ..., a_n+1 of linearly spaced intermediate points
    linear_seq = []
    linear_seq.append(a)
    for i in range(n):
        delta = (i + 1) / (n + 1) * (b - a)
        linear_seq.append(a + delta)
    linear_seq.append(b)
    
    convergence = True
    while convergence is True:
        convergence = False
        
        for i, alpha in enumerate(linear_seq[1:-1]):
            index = i + 1
            current_pt = alpha
            prev_pt = linear_seq[index - 1]
            next_pt = linear_seq[index + 1]
            
            # arc length integrals
            length_1 = integrate.quad(integrand, 0, 1, args=(prev_pt, current_pt))[0] 
            length_2 = integrate.quad(integrand, 0, 1, args=(current_pt, next_pt))[0] 
            
            radius = 2. * max(np.linalg.norm(current_pt - prev_pt), np.linalg.norm(next_pt - current_pt))
            samples = np.random.uniform(-radius, radius, size=(m, base_dim)) + alpha # radius ball centered at alpha
            min_dist = length_1 + length_2
            best_pt = current_pt

            # find min intermediate path distance and update values
            for sample_pt in samples:
                length_1 = integrate.quad(integrand, 0, 1, args=(prev_pt, current_pt))[0]
                length_2 = integrate.quad(integrand, 0, 1, args=(current_pt, next_pt))[0] 
                sample_dist = length_1 + length_2
                if sample_dist < min_dist:
                    min_dist = sample_dist
                    best_pt = sample_pt
            linear_seq[i] = best_pt

            # if any updates aren't less than the tolerance, continue while loop
            if not np.linalg.norm(current_pt - best_pt) < tol:
                convergence = True
                
    total_dist = 0
    for i in range(len(linear_seq)-1):
        path_dist = integrate.quad(integrand, 0, 1, args=(linear_seq[i], linear_seq[i+1]))[0]
        total_dist += path_dist

    return total_dist


def test_approximations(n_samples, parameter_n_range):
    """ Test approximation algorithm against closed form hyperbolic distances """
    s = np.random.uniform(-2, 2, size=(n_samples, 2))
    r = np.random.uniform(-2, 2, size=(n_samples, 2))
    result_dict = {}
    for n in parameter_n_range:
        errors = []
        for i in range(n_samples):
            x = hyperboloid_map(r[i])
            y = hyperboloid_map(s[i])
            true_dist = hyperboloid_dist(x, y)
            errors.append(approximate_distance(s[i], r[i], integrand_hyperboloid, n=n, m=10) / true_dist)
        result_dict[n] = errors
    return result_dict