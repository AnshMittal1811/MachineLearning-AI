import numpy as np
import networkx as nx
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def minkowski_dist(x, y, c=1):
    """Minkowski separation between x and y, c = speed of light parameter"""
    d0 = -c**2 * (x[0] - y[0])**2
    d1 = np.sum((x[1:] - y[1:])**2)
    return d0 + d1

def add_edges(X, c=1):
    """Predict edges from embedding coordinates in Minkowski spacetime, add directed edge
    if nodes have corresponding spacetime coordinates that are time-separated
    
    Parameters: X - N x d array with Minkowski spacetime coordinates
    Return: edges - list of directed edges (i, j)
    """
    n = X.shape[0]
    edges = []
    for i in range(n):
        for j in range(i, n):
            if minkowski_dist(X[i], X[j], c) < 0:
                if X[i][0] < X[j][0]:
                    edges.append((i, j))
                else:
                    edges.append((j, i))
    return edges

# define evaluation metrics to test edge predictions in graph recreation
def sensitivity(G, pred_E):
    # True positive rate
    E = list(G.edges())
    return len([x for x in E if x in pred_E]) / len(E)

def specificity(G, pred_E):
    # False positive rate = 1 - specificity
    A = nx.adjacency_matrix(G).toarray()
    non_edges = [tuple(x) for x in np.argwhere(A == 0)]
    n = A.shape[0]
    node_list = list(G.nodes())
    pred_A = np.zeros((n, n))
    for edge in pred_E:
        i = node_list.index(edge[0])
        j = node_list.index(edge[1])
        pred_A[i][j] = 1
    pred_non_edges = [tuple(x) for x in np.argwhere(pred_A == 0)]
    return len([x for x in non_edges if x in pred_non_edges]) / len(non_edges)

def roc_auc_score(G, X_embed):
    tpr = []
    fpr = []
    node_list = list(G.nodes())
    for c in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
        pred_E = add_edges(X_embed, c)
        pred_E = [(node_list[e[0]], node_list[e[1]]) for e in pred_E]
        tpr.append(sensitivity(G, pred_E))
        fpr.append(1 - specificity(G, pred_E))
    tpr.append(1)
    fpr.append(1)
    return auc(fpr, tpr)

def plot_roc_curve(G, X_embed, title='ROC Curve'):
    tpr = []
    fpr = []
    node_list = list(G.nodes())
    for c in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
        pred_E = add_edges(X_embed, c)
        pred_E = [(node_list[e[0]], node_list[e[1]]) for e in pred_E]
        tpr.append(sensitivity(G, pred_E))
        fpr.append(1 - specificity(G, pred_E))
    tpr.append(1)
    fpr.append(1)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=3, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr, tpr, color='black')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right", edgecolor='black', fontsize=12)
    plt.show();