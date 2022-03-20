import numpy as np
import networkx as nx
from eval_metrics import roc_auc_score

def minkowski_dist(x, y, c=1):
    """Minkowski separation between x and y, c = speed of light parameter"""
    d0 = -c**2 * (x[0] - y[0])**2
    d1 = np.sum((x[1:] - y[1:])**2)
    return d0 + d1

def LorentzMDS(M, d):
    """Performs Lorentzian Multidimensional Scaling
    Parameters
    ----------
    M : Minkowski separation matrix (n x n)
    d: Dimension of the output configuration
    
    Returns
    -------
    X : Matrix with columns as the output embedding vectors (n x d)
    """
    # get shape of distance matrix                                                                         
    n = M.shape[0]
    
    # centering matrix
    C = np.identity(n) -(1/n)*np.ones((n,n))
    
    # compute gram matrix                                                                                    
    B = -(1/2)*C.dot(M).dot(C)
    
    # solve for eigenvectors and eigenvalues and sort descending                                                   
    w, v = np.linalg.eigh(B)                                                  
    idx = np.argsort(w)[::-1]
    eigvals = w[idx]
    eigvecs = v[:,idx]
    
    # select d-1 largest positive eigenvalues/eigenvectors and most negative eigenvalue/eigenvector
    S  = np.diag(np.concatenate((np.sqrt(-1*eigvals[-1:]), np.sqrt(eigvals[:d-1]))))
    U  = np.hstack((eigvecs[:,-1:], eigvecs[:,:d-1]))
    X  = np.dot(S, np.transpose(U))
    X = np.transpose(X)

    return X

class MinkowskiEmbedding():
    """
    Given a directed acyclic graph, compute Minkowski separations and apply
    Lorentzian Multidimensional Scaling to find spacetime embedding coordinates. 
    
    API design is modeled on the standard scikit-learn Classifier API
    """
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        
    # longest path distance between s, t
    def longest_path(self, G, s, t):
        """Find longest directed path from node s to node t

        Parameters: G - directed acyclic graph, s - source node, t - target node
        Return: int - num. edges in longest path
        """
        if s!= t and nx.has_path(G, source=s, target=t):
            path_lengths = [len(x) for x in list(nx.all_simple_paths(G, source=s, target=t))]
            return max(path_lengths)
        else:
            return 0
        
    # get longest path lengths for all time-separated nodes
    def longest_paths(self, G):
        """Find longest directed paths between all time-separated nodes

        Parameters: G - directed acyclic graph
        Return: N x N matrix with pairwise longest paths - L_ij^2
        """
        n = len(list(G.nodes()))
        node_list = list(G.nodes())
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i][j] = -self.longest_path(G, node_list[i], node_list[j])**2
        # let longest path separation matrix be symmetric
        self.time_separations = M + np.transpose(M)
    
    # get naive spacelike distances if no path i --> j
    def naive_spatial_distances(self, G, max_dist=None):
        """Find pairwise space-like separations between disconnected nodes

        Parameters: L - NxN time-like separation matrix, G - directed acyclic graph
        Return: M - NxN minkowski separation matrix
        """
        M = self.time_separations
        n = M.shape[0]
        node_list = list(G.nodes())
        if max_dist == None:
            max_dist = np.max(np.abs(M))
        for i in range(n):
            for j in range(i, n):
                if  M[i,j] == 0 and (i != j):
                    past_i = nx.ancestors(G, node_list[i])
                    past_j = nx.ancestors(G, node_list[j])
                    common_past = list(past_i.intersection(past_j))
                    future_i = nx.descendants(G, node_list[i])
                    future_j = nx.descendants(G, node_list[j])
                    common_future = list(future_i.intersection(future_j))
                    naive_dists = []
                    if (len(common_past) > 0) and (len(common_future) > 0):
                        for k in common_past:
                            for l in common_future:
                                L_kl = self.longest_path(G, k, l)
                                if L_kl != 0:
                                    naive_dists.append(L_kl)
                                else:
                                    naive_dists.append(max_dist)
                        S_ij = np.min(naive_dists)**2
                        M[i,j] = S_ij
                        M[j,i] = S_ij
                    else:
                        M[i,j] = max_dist
                        M[j,i] = max_dist
        self.minkowski_separations = M
    
    def fit(self, G, max_dist=None):
        """
        Compute Minkowski separation matrix, apply Lorentzian MDS from the graph,
        and return the embedding coordinates.
        Parameters
        ----------
        G : Input graph
        max_dist: maximum spacelike separation value, default = length of longest path
        c: speed of light parameter
        
        Return: embedding coordinates as Nxd array
        """
        # compute minkowski separation matrix
        self.node_list = list(G.nodes())
        self.n = len(self.node_list)
        self.longest_paths(G)
        self.naive_spatial_distances(G, max_dist=max_dist)
        X = LorentzMDS(self.minkowski_separations, d=self.n_components)
        self.embedding = X
        # check for possible time coordinate reversal in embedding
        if roc_auc_score(G, self.embedding) < 0.5:
            # reflect embedding coordinates over timelike dimension
            self.embedding[:, 0] = -self.embedding[:, 0]
        return self.embedding
    
    def predict(self, G, c=1):
        """
        Predict edges from embedding coordinates in Minkowski spacetime, add directed edge
        if nodes have corresponding spacetime coordinates that are time-separated.
        
        Parameters
        ----------
        G : Input graph
        c: speed of light parameter
        
        Return: edges - list of directed edges (i, j), **with labels given in original G (i.e. node '97100014')
        """
        X = self.embedding
        n = X.shape[0]
        edges = []
        node_list = list(G.nodes())
        for i in range(n):
            for j in range(i, n):
                if minkowski_dist(X[i], X[j], c) < 0:
                    if X[i][0] < X[j][0]:
                        edges.append((node_list[i], node_list[j]))
                    else:
                        edges.append((node_list[j], node_list[i]))
        return edges