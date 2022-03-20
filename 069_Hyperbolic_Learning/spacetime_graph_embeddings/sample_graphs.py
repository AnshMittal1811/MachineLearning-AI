import numpy as np
import networkx as nx

def minkowski_dist(x, y, c=1):
    """Minkowski separation between x and y, c = speed of light parameter"""
    d0 = -c**2 * (x[0] - y[0])**2
    d1 = np.sum((x[1:] - y[1:])**2)
    return d0 + d1

def minkowski_interval(N, D):
    """ Sample N points from minkowski interval [a,b]^D, where each sample point is in causal
    future of endpoint a and causal past of endpoint b (page 4 from https://arxiv.org/abs/1408.1274)
    
    Parameters: N: observations, D: minkowski dimension """
    R = np.random.random((N, D))
    a = np.concatenate(([0], np.zeros(D-1)+0.5))
    b = np.concatenate(([1], np.zeros(D-1)+0.5))
    R[0] = a
    R[1] = b
    for i in range(2, N):
        while (minkowski_dist(a, R[i, :]) > 0) or ((minkowski_dist(R[i, :], b) > 0)):
            R[i, :] = np.random.random(D)
    return R

def uniform_box_graph(n, d=2):
    """Sample N points from minkowski interval [0,1]^D and connect timelike coordinates
    with directed edge from past to future
    
    Parameters: n observations, d dimension
    Return: Directed acyclic graph G
    """
    X = np.random.random((n, d))
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, position=X[i])
        for j in range(i, n):
            if minkowski_dist(X[i], X[j]) < 0:
                if X[i][0] < X[j][0]:
                    G.add_edge(i, j)
                else:
                    G.add_edge(j, i)
    return G

def causal_set_graph(n, d=2):
    """Sample N points from minkowski interval [a,b]^D and connect timelike coordinates
    with directed edge from past to future
    
    Parameters: n observations, d dimension
    Return: Directed acyclic graph G
    """
    X = minkowski_interval(n, d)
    G = nx.DiGraph()
    n = X.shape[0]
    for i in range(n):
        G.add_node(i, position=X[i])
        for j in range(i, n):
            if minkowski_dist(X[i], X[j]) < 0:
                if X[i][0] < X[j][0]:
                    G.add_edge(i, j)
                else:
                    G.add_edge(j, i)
    return G

def random_dag(n, p=0.1):
    if p < 0 or p > 1:
        print('Error: probability param not in [0,1])
        return
    G = nx.erdos_renyi_graph(n, p=p, seed=None, directed=True)
    return G

