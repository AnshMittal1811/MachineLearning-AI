import numpy as np
from scipy.optimize import minimize
import sys
my_path = 'C:\\Users\\dreww\\Desktop\\manifold-metric-learn' # path to utils folder
sys.path.append(my_path)
from manifold_distances import *
from manifold_instances import *

def similar_pairs(labels):
    """ Generate similarity/dissimilarity pair sets based on indices with the same label """
    P = [] # set of similar pairs
    Q = [] # set of dissimilar pairs
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] == labels[j]:
                P.append([i, j])
            else:
                Q.append([i, j])
    return P, Q

def mmc_loss(L, B, labels, manifold_map_fn, integrand, manifold_dist_fn=None, C=0.1, scaling_penalty=10):
    """
    Function to calculate the MMC loss for generalized manifold surfaces
    Parameters
    ----------
    L : optimization argument we seek to find, linear transformation matrix L applied to B
    B : set of m points b1, ..., bm in Euclidean base space
    labels : set of m classification labels y1, ..., ym
    manifold_map_fn : function F that maps points in B to manifold S
    manifold_dist_fn : explicit distance function on the manifold, if none then approximate
    integrand: arc length formula for piecewise linear distance approximation
    C: constant parameter controlling degree of push/pull in optimization routine 
    """
    
    # to use explicit manifold distance formulas, need to get manifold points by applying F to B
    dim = len(B[0])
    L = L.reshape(dim, dim)
    X = []
    for b in B:
        X.append(manifold_map_fn(np.matmul(L, b)))
    
    # get similarity pair indices
    similar_ix, dissimilar_ix = similar_pairs(labels)
    
    # get first summation term
    pull = 0
    for i, j in similar_ix:
        if manifold_dist_fn == None: # approximation method
            d_ij = approximate_distance(np.matmul(L, B[i]), np.matmul(L, B[j]), integrand=integrand, n=3, m=1, tol=0.01)
        else:
            d_ij = manifold_dist_fn(X[i], X[j])
        pull += (1 - C) * d_ij / len(similar_ix)
#        print('Similar pair {}, {}'.format(i, j))
            
    # get second summation term
    push = 0
    for i, j in dissimilar_ix:
        if manifold_dist_fn == None: # approximation method
            d_ij = approximate_distance(np.matmul(L, B[i]), np.matmul(L, B[j]), integrand=integrand, n=3, m=1, tol=0.01)
        else:
            d_ij = manifold_dist_fn(X[i], X[j])
        push += C * d_ij / len(dissimilar_ix)
    
    loss = pull - push + scaling_penalty*(np.matmul(L.T, L).sum())
    return loss

def mmc_optimize(B, labels, manifold_map_fn, integrand, solver='Powell', max_eval=50, summary=False):
    """
    Wrapper for scipy optimization routine, return linear transformation matrix L that minimizes mmc loss
    Parameters
    ----------
    B : set of m points b1, ..., bm in Euclidean base space
    labels : set of m classification labels y1, ..., ym
    manifold_map_fn : function F that maps points in B to manifold S
    integrand: arc length formula for piecewise linear distance approximation
    solver: scipy minimization method (Powell, Nelder-Mead, Newton-CG)
    max_eval: limit on number of loss function evaluations
    """
    dim = len(B[0])
    L_init = np.eye(dim)
    manifold_opt_L = minimize(mmc_loss, L_init,
                          args=(B, labels, manifold_map_fn, integrand),
                          method=solver, options={'disp': True, 'maxfev': max_eval})
    if summary:
        print(manifold_opt_L)
    L_new = manifold_opt_L.x.reshape(dim,dim)
    return L_new
