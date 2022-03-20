# import libraries
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances

# ----------------------------------------------------------
# ----- SIMPLE HELPER FUNCTIONS (WITH COMPLEX NUMBERS) -----
# ----------------------------------------------------------

# define helper functions
def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

# convert array or list [a, b] to complex number a+bi
def to_complex(xi):
    return np.complex(xi[0], xi[1])

# initialize embedding configuration
def init_z(n, dim=2, low=-0.5, high=0.5, complex_num = True):
    random_config = np.random.uniform(low, high, size=(n, dim))
    z_config = np.array([to_complex(x) for x in random_config])
    if complex_num:
        return z_config
    else:
        return random_config

# sample uniform random within circle
def generate_data(n, radius=0.3):
    theta = np.random.uniform(0, 2*np.pi, n)
    u = np.random.uniform(0, radius, n)
    r = np.sqrt(u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    init_data = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    return init_data

# optionally clip data points within disk boundary
def proj(theta, eps=1e-2):
    if norm(theta) >= 1:
        theta = theta/norm(theta) - eps
    return theta

# alternate poincare distance formula with complex numbers
def poincare_dist(zi, zj):
    if not isinstance(zi,complex):
        zi = to_complex(zi)
    if not isinstance(zj,complex):
        zj = to_complex(zj)
    return 2*np.arctanh(norm(zi - zj) / norm(1 - zi*np.conj(zj)))

# compute symmetric poincare distance matrix
def pd_matrix(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = poincare_dist(embedding[i], embedding[j])
    return dist_matrix

# --------------------------------------------------
# ----- LOSS FUNCTIONS AND COMPUTING GRADIENTS -----
# --------------------------------------------------
    
# partial derivative of poincare distance
def partial_d(xi, xj):
    if not isinstance(xi,complex):
        xi = to_complex(xi)
    if not isinstance(xj,complex):
        xj = to_complex(xj)
    v1 = np.real(xi) - np.real(xj)
    v2 = np.imag(xi) - np.imag(xj)
    v3 = np.real(xi)*np.real(xj) + np.imag(xi)*np.imag(xj) - 1
    v4 = np.real(xi)*np.imag(xj) - np.real(xj)*np.imag(xi)
    t = np.sqrt((v1**2 + v2**2) / (v3**2 + v4**2))
    dxi_1 = 2*t / (1 - t**2) * (v1 / (v1**2 + v2**2) - (np.real(xj)*v3 + np.imag(xj)*v4) / (v3**2 + v4**2))
    dxi_2 = 2*t / (1 - t**2) * (v2 / (v1**2 + v2**2) - (np.real(xi)*v4 - np.imag(xj)*v3) / (v3**2 + v4**2))
    return np.array([dxi_1, dxi_2])

def compute_gradients(Z, dissimilarities, alpha=1):
    n = Z.shape[0]
    gradients = np.zeros((n, 2))
    for i in range(n):
        grad_zi = 0
        for j in range(i+1, n):
            dd_loss = 2*poincare_dist(Z[i], Z[j]) - 2*alpha*dissimilarities[i][j]
            dd_dist = partial_d(Z[i], Z[j])
            grad_zi += dd_loss * dd_dist
        if norm(grad_zi) > 1:
            grad_zi = grad_zi / norm(grad_zi)
        gradients[i] = grad_zi
    return gradients

def loss_fn(embed_config, dissimilarities, alpha=1):
    n = dissimilarities.shape[0]
    loss = 0
    for i in range(n):
        for j in range(i+1, n):
            zi_error = (poincare_dist(embed_config[i], embed_config[j]) - alpha*dissimilarities[i][j])**2
            loss += zi_error
    return loss

#----------------------------------
#----- Hyperbolic Line Search -----
#----------------------------------
    
def step_error(r, Z, g, dissimilarities, n=None):
    if n == None:
        n = dissimilarities.shape[0]
    if not isinstance(Z[0], complex):
        Z = np.array([to_complex(zi) for zi in Z])
    if not isinstance(g[0], complex):
        g = np.array([to_complex(gi) for gi in g])
    M_r = []
    for j in range(n):
        M_r.append((-r*g[j] + Z[j]) / (-r*g[j] * np.conj(Z[j]) + 1))
    return loss_fn(np.array(M_r), dissimilarities)

# approximate hyperbolic line search
def line_search(Z, dissimilarities, g, n, r0, rmax, verbose=False):
    if not isinstance(Z[0], complex):
        Z = np.array([to_complex(zi) for zi in Z])
    if not isinstance(g[0], complex):
        g = np.array([to_complex(gi) for gi in g])
    Z_norm = np.array([norm(z)**2 for z in Z])
    M_prime = g*Z_norm
    qprime_0 = np.dot(np.real(M_prime).T, np.real(g)) + np.dot(np.imag(M_prime).T, np.imag(g))
    p = 0.5
    r = r0
    q0 = step_error(0, Z, g, dissimilarities, n)
    roof_fn = lambda r: q0 + p*qprime_0*r
    rmin = 1e-10
    while r < rmax and step_error(r, Z, g, dissimilarities, n) < roof_fn(r):
        if verbose:
            print('step size: ',r)
            print('roof fn: ', roof_fn(r))
            print('step error: ', step_error(r, Z, g, dissimilarities, n))
        r = 2*r
    while r > rmax or step_error(r, Z, g, dissimilarities, n) > roof_fn(r):
        if verbose:
            print('step size: ',r)
            print('roof fn: ', roof_fn(r))
            print('step error: ', step_error(r, Z, g, dissimilarities, n))
        if r < rmin:
            return 2*r
        r = r/2
    return r

#--------------------
#----- HyperMDS -----
#--------------------

class HyperMDS():
    
    def __init__(self, dim=2, max_iter=3, verbose=0, eps=1e-5, alpha=1, save_metrics=False,
                 random_state=None, dissimilarity="euclidean"):
        self.dim = dim
        self.dissimilarity = dissimilarity
        self.max_iter = max_iter
        self.alpha = alpha
        self.eps = eps
        self.verbose = verbose
        self.random_state = random_state
        self.save_metrics = save_metrics
        if self.save_metrics:
            self.gradient_norms = []
            self.steps = []
            self.rel_steps = []
            self.losses = []
        
    def init_embed(self, low=-0.2, high=0.2, complex_num=True):
        random_config = np.random.uniform(low, high, size=(self.n, self.dim))
        c_config = np.array([to_complex(x) for x in random_config], dtype=np.complex_)
        if complex_num:
            self.embedding = c_config
        else:
            self.embedding = random_config
    
    def loss_fn(self):
        loss = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                d_ij = poincare_dist(self.embedding[i], self.embedding[j])
                delta_ij = self.alpha*self.dissimilarity_matrix[i][j]
                loss += (d_ij - delta_ij)**2
        self.loss = loss

    def compute_gradients(self):
        gradients = np.zeros((self.n, 2))
        for i in range(self.n):
            grad_zi = 0
            for j in range(i+1, self.n):
                delta_ij = self.alpha*self.dissimilarity_matrix[i][j]
                dd_ij = 2*poincare_dist(self.embedding[i], self.embedding[j])
                ddelta_ij = 2*delta_ij
                dd_loss = dd_ij - ddelta_ij
                dd_dist = partial_d(self.embedding[i], self.embedding[j])
                weight_ij = 2 / (self.n * (self.n - 1)) #* 1/(delta_ij**2)
                grad_zi += dd_loss * dd_dist * weight_ij
            #if norm(grad_zi) > 1:
            #    grad_zi = grad_zi / norm(grad_zi)
            gradients[i] = grad_zi
        self.gradients = gradients
    
    def fit(self, X, init=None):
        """
        Uses gradient descent to find the embedding configuration in the Poincaré disk
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        init: initial configuration of the embedding coordinates
        """
        self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X, init=None, init_low=-0.2, init_high=0.2, smax=10, 
                      max_epochs = 40, verbose=False):
        """
        Fit the embedding from X, and return the embedding coordinates
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        init: initial configuration of the embedding coordinates
        init_low: lower bound for range of initial configuration
        init_high: upper bound for range of initial configuration
        smax: max distance window to seek embedding
        max_epochs: maximum number of gradient descent iterations
        verbose: optionally print training scores
        """
        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix = euclidean_distances(X)
        elif self.dissimilarity == 'hyperbolic':
            self.dissimilarity_matrix = pd_matrix(X)
        self.n = self.dissimilarity_matrix.shape[0]
        
        # set initial config
        self.init_embed(low=init_low, high=init_high)
        
        # set a max distance window for embedding
        smax = smax
        prev_loss = np.inf
        for i in range(max_epochs):
            # break if loss decrease < tolerance
            self.loss_fn()
            if (prev_loss - self.loss) / prev_loss < 0.0001:
                break
            prev_loss = self.loss
            self.compute_gradients()
            # set max step size and perform line search
            rmax = 1/(norm(self.gradients, axis=1).max() +self.eps) * np.tanh(smax/2)
            r = line_search(self.embedding, self.dissimilarity_matrix, self.gradients,
                            self.n, 1, rmax)
            if r < 1e-8:
                break
            # update each zi
            for j in range(self.n):
                zi_num = -r*to_complex(self.gradients[j]) + self.embedding[j]
                zi_denom = -r*to_complex(self.gradients[j]) * np.conj(self.embedding[j]) + 1
                zi_prime = zi_num / zi_denom
                self.embedding[j] = zi_prime
            # optionally save training metrics
            if self.save_metrics:
                self.gradient_norms.append(norm(self.gradients, axis=1).max())
                self.steps.append(r)
                self.rel_steps.append(r/rmax)
                self.losses.append(self.loss)
            if verbose:
                print('Epoch ' + str(i+1) + ' complete')
                print('Loss: ', self.loss)
                print('\n')
                
        # final loss value
        self.loss_fn()
        # remove complex numbers in embedding
        final_emb = self.embedding.reshape(-1,1)
        self.embedding = np.hstack((np.real(final_emb), np.imag(final_emb)))
        return self.embedding
    
#----------------------------------------
#----- EVALUATION UTILITY FUNCTIONS -----
#----------------------------------------
        
# compute Sammon stress of the embedding
def sammon_stress(embedding, dissimilarity_matrix, alpha=1):
    stress = 0
    scale = 0
    n = embedding.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if dissimilarity_matrix[i][j] != 0:
                delta_ij = alpha * dissimilarity_matrix[i][j]
                d_ij = poincare_dist(embedding[i], embedding[j])
                scale += delta_ij
                stress += (d_ij - delta_ij)**2 / delta_ij
    return stress/scale