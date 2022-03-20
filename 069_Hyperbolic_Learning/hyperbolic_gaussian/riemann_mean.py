# import libraries
import numpy as np
import sys

# import modules within repository
sys.path.append('C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\utils') # path to utils folder
sys.path.append('C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\hyperbolic_gaussian')
from utils import *
from distributions import *

#-------------------------------------------------------------------
#----- Riemannian Barycenter Optimization in Hyperboloid Model -----
#-------------------------------------------------------------------

def exp_map(v, theta_k, eps=1e-6):
    """ Exponential map that projects tangent vector v onto hyperboloid"""
    # v: vector in tangent space at theta_k
    # theta: parameter vector in hyperboloid with centroid coordinates
    # project vector v from tangent minkowski space -> hyperboloid"""
    return np.cosh(norm(v))*theta_k + np.sinh(norm(v)) * v / (norm(v) + eps)

def minkowski_distance_gradient(u, v, eps=1e-5):
    """ Riemannian gradient of hyperboloid distance w.r.t point u """ 
    # u,v in hyperboloid
    dot = np.sqrt(minkowski_dot(u,v)**2 - 1)
    if np.isnan(dot):
        dot = eps
    return -1*dot * v

def minkowski_loss_gradient(theta_k, X, w, eps=1e-5):
    """ Riemannian gradient of error function w.r.t theta_k """
    # X : ALL data x1, ..., xN (not just within clusters like K-means) - shape N x 1
    # theta_k: point in hyperboloid at cluster center
    # w: vector with weights w_1k, ..., w_Nk - shape N x 1
    # returns gradient vector
    weighted_distances = w*np.array([-1*hyperboloid_dist(theta_k, x, metric='minkowski') for x in X]) # scalars
    distance_grads = np.array([minkowski_distance_gradient(theta_k, x) for x in X]) # list of vectors
    grad_loss = np.array([weighted_distances[i]*distance_grads[i] for i in range(len(weighted_distances))]) # list of vectors
    grad_loss = 2*np.sum(grad_loss, axis=0) # vector
    #grad_loss = grad_loss / np.max(grad_loss)
    if np.isnan(grad_loss).any():
        return np.array([eps, eps, eps])
    else:
        return grad_loss

def project_to_tangent(theta_k, minkowski_grad):
    """ 
    Projects vector in ambient space to hyperboloid tangent space at theta_k 
    Note: returns our hyperboloid gradient of the error function w.r.t theta_k
    """
    # minkowski_grad: riemannian gradient vector in ambient space
    # theta_k: point in hyperboloid at cluster center
    return minkowski_grad + minkowski_dot(theta_k, minkowski_grad)*theta_k

def update_step(theta_k, hyperboloid_grad, alpha=0.1):
    """ 
    Apply exponential map to project the gradient and obtain new cluster center
    Note: returns updated theta_k
    """
    # theta_k: point in hyperboloid at cluster center
    # hyperboloid_grad: hyperboloid gradient in tangent space
    # alpha: learning rate > 0
    new_theta_k = exp_map(-1*alpha*hyperboloid_grad, theta_k)
    return new_theta_k

def barycenter_loss(theta_k, X, w):
    """ Evaluate barycenter loss for a given gaussian cluster """
    # X : ALL data x1, ..., xN (not just within clusters like K-means) - shape N x 1
    # theta_k: parameter matrix with cluster center points - 1 x n
    # w: weights w_1k, ..., w_Nk - shape N x 1
    distances = np.array([hyperboloid_dist(theta_k, x, metric='minkowski')**2 for x in X])
    weighted_distances = w * distances
    loss = np.sum(weighted_distances)
    return loss

def overall_loss(theta, X, W):
    """ Evaluate barycenter loss for a given gaussian cluster """
    # X : ALL data x1, ..., xN (not just within clusters like K-means) - shape N x 1
    # theta: parameter matrix with cluster center points - k x n
    # W: matrix with weights w_1k, ..., w_Nk - shape k x N
    loss = 0
    K = W.shape[1]
    for i in range(K):
        distances = np.array([hyperboloid_dist(theta[i], x)**2 for x in X])
        weighted_distances = W[i, :] * distances
        loss += np.sum(weighted_distances)
    return loss

def weighted_barycenter(theta_k, X, w, num_rounds = 10, alpha=0.3, tol = 1e-8, verbose=False):
    """ Estimate weighted barycenter for a gaussian cluster with optimization routine """
    # X : ALL data x1, ..., xN (not just within clusters like K-means) - shape N x 1
    # theta_k: parameter matrix with cluster center points - k x n
    # w: weights w_1k, ..., w_Nk - shape N X 1
    # num_rounds: training iterations
    # alpha: learning rate
    # tol: convergence tolerance, exit if updates smaller than tolerance
    centr_pt = theta_k.copy()
    centr_pts = []
    losses = []
    for i in range(num_rounds):
        gradient_loss = minkowski_loss_gradient(centr_pt, X, w)
        tangent_grad = project_to_tangent(centr_pt, -gradient_loss)
        new_centr = update_step(centr_pt, tangent_grad, alpha=alpha)
        if np.isnan(new_centr).any() or np.isinf(new_centr).any():
            new_centr = update_step(centr_pt, tangent_grad, alpha=alpha/10)
            if np.isnan(new_centr).any() or np.isinf(new_centr).any():
                break
        centr_pt = new_centr
        centr_pts.append(centr_pt)
        losses.append(barycenter_loss(centr_pt, X, w))
        if verbose:
            print('Epoch ' + str(i+1) + ' complete')
            print('Loss: ', barycenter_loss(centr_pt, X, w))
            print('\n')
        #if hyperboloid_dist(centr_pts[i+1], centr_pts[i], metric='minkowski') < tol:
        #    break
    return centr_pt
