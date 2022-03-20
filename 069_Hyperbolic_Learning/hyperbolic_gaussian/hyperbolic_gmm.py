# import libraries
import numpy as np
import sys
import os

# import modules within repository
sys.path.append('C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\utils') 
sys.path.append('C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\hyperbolic_gaussian')
from utils import *
from riemann_mean import *
from distributions import *

#----------------------------------------------------------
#----- Gaussian Mixture Model in Hyperboloid Space --------
#----------------------------------------------------------

class HyperbolicGMM():
    """
    Gaussian Mixture Model in hyperbolic space where we use the Wrapped Normal Distribution
    and its p.d.f to determine likelihood of cluster assignments. Applies gradient descent in
    the hyperboloid model to iteratively compute the Riemannian barycenter.
    
    Note: Follows the Expectation-Maximization (EM) approach for Unsupervised Clustering
    """
    
    def __init__(self, n_clusters=3, verbose=False):
        """ Initialize Gaussian Mixture Model and set training parameters """
        self.n_clusters = n_clusters
        self.verbose = verbose
        
    def init_params(self, X=None, radius=0.15, method='random'):
        """ Initialize parameter configurations that define gaussian clusters
        Options: 
            1.) Randomly sample points around small uniform ball
            2.) Set cluster center at randomly chosen data points
            3.) Initialize with K-Means clustering (not implemented here)
        """
        if method == 'select' and X is not None:
            indices = np.random.randint(0, self.n_samples, self.n_clusters)
            self.means = X[indices]
            
        else:
            theta = np.random.uniform(0, 2*np.pi, self.n_clusters)
            u = np.random.uniform(0, radius, self.n_clusters)
            r = np.sqrt(u)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            centers = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
            self.means = poincare_pts_to_hyperboloid(centers, metric='minkowski')
            
        # methods exist to initialize variances but I use unit variance here
        self.variances = np.tile([1, 1], self.n_clusters).reshape((self.n_clusters, 2))
        
        # we initialize the cluster weights evenly
        self.cluster_weights = np.repeat(1/self.n_samples, self.n_clusters)
        
    #------------------------------------------------------------------------------
    #------------------------ EXPECTATION STEP ------------------------------------
    #------------------------------------------------------------------------------

    def update_likelihoods(self, X):
        """ Compute likelihoods using log-pdf of Wrapped Normal Distribution """
        W = np.zeros((self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            for j in range(self.n_clusters):
                W[i, j] = self.cluster_weights[j] * np.exp((log_pdf(z=X[i], mu=self.means[j], sigma=self.variances[j])))
            # divide by row sum to normalize assignment probabilities
            row_sum = W[i, :].sum()
            W[i, :] = W[i, :] / row_sum
        # set assignment probabilities to be updated
        self.likelihoods = W
                
    #-----------------------------------------------------------------------------
    #------------------------ MAXIMIZATION STEP ----------------------------------
    #-----------------------------------------------------------------------------
    
    def update_cluster_weights(self):
        """ Update new cluster weights based on cluster assignment likelihoods """
        for j in range(self.n_clusters):
            new_weight = self.likelihoods[:, j].sum() / self.n_samples
            self.cluster_weights[j] = new_weight
        
    def update_means(self, X, num_rounds=10, alpha=0.3, tol=1e-4):
        """ Apply weighted barycenter algorithm to update gaussian clusters """
        train_means = []
        for i in range(self.n_clusters):
            new_mean = weighted_barycenter(self.means[i], X, self.likelihoods[:, i], num_rounds = num_rounds, alpha=alpha, tol=tol)
            train_means.append(new_mean)
        self.means = np.array(train_means)
        
        
    #-----------------------------------------------------------------------------
    #------------------------ TRAINING ROUTINE -----------------------------------
    #-----------------------------------------------------------------------------
    
    
    def loss_fn(self, X):
        """ Wrapper for auxilliary function to compute total barycenter loss """
        loss = 0
        for i in range(self.n_clusters):
            distances = np.array([hyperboloid_dist(-self.means[i], x, metric='minkowski')**2 for x in X])
            weighted_distances = self.likelihoods[:, i] * distances
            loss += np.sum(weighted_distances / np.max(weighted_distances))
        self.loss = loss
            
    def fit(self, X, y=None, max_epochs=40, alpha=0.3, metrics=False, verbose=False, init_means='random'):
        """
        Fit K gaussian distributed clusters to data, return cluster assignments by max likelihood 
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y: optionally train a supervised model with given labels y (in progress)
        max_epochs: maximum number of gradient descent iterations
        verbose: optionally print training scores
        """
        
        # make sure X within poincaré ball
        #if (norm(X, axis=1) > 1).any():
        #    X = X / (np.max(norm(X, axis=1)))
        
        # initialize gaussian mixture parameters
        self.n_samples = X.shape[0]
        self.init_params(X, method=init_means)
        if metrics:
            train_means = []
            train_ll = []
            train_losses = []
            
        # initialize assignments as the most likely cluster for each xi
        self.assignments = np.zeros((self.n_samples, self.n_clusters))
        if verbose:
            self.update_likelihoods(X)
            self.loss_fn(X)
            print('Initial Loss: ' + str(self.loss))
            
        # loop through the expectation and maximization steps
        for j in range(max_epochs):
            # update likelihoods given new parameters
            self.update_likelihoods(X)
            self.update_cluster_weights()
            # update parameters given likelihoods
            self.update_means(X, num_rounds=10, alpha=alpha, tol = 1e-8)
            self.loss_fn(X)
            if metrics:
                train_means.append(self.means)
                train_ll.append(self.likelihoods)
                train_losses.append(self.loss)
            if verbose:
                print('---- Epoch ' + str(j) + ' complete ---- Loss: ' + str(self.loss))

        # assign final probabilities and make assignments
        self.update_likelihoods(X)
        for i in range(self.n_samples):
            # zero out current cluster assignment
            self.assignments[i, :] = np.zeros((1, self.n_clusters))
            # find centroid that gives maximum likelihood
            k_max = np.argmax(self.likelihoods[i, :])
            self.assignments[i, k_max] = 1
                
        self.labels = np.argmax(self.likelihoods, axis=1)
        self.loss_fn(X)
        if metrics:
            self.train_metrics = {'means': np.array(train_means),
                                  'likelihoods': np.array(train_ll),
                                  'losses': np.array(train_losses)}
                                  