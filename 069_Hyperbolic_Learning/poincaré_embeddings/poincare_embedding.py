# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:59:31 2019

@author: dreww
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)

class PoincareModel():
    
    def __init__(self, relations, n_components=2, eta=0.1, n_negative=10,
                 eps=1e-5, burn_in=10, burn_in_eta=0.01, init_lower=-0.001,
                 init_upper=0.001, dtype=np.float64, seed=0):
        self.relations = relations
        self.n_components = n_components
        self.eta = eta  # Learning rate for training
        self.burn_in_eta = burn_in_eta  # Learning rate for burn-in
        self.n_negative = n_negative
        self.eps = eps
        self.burn_in = burn_in
        self.dtype = dtype
        self.init_lower = init_lower
        self.init_upper = init_upper
       
    def init_embeddings(self):
        unique_nodes = np.unique([item for sublist in self.relations for item in sublist])
        theta_init = np.random.uniform(self.init_lower, self.init_upper, 
                                       size=(len(unique_nodes), self.n_components))
        embedding_dict = dict(zip(unique_nodes, theta_init))
        self.nodes = unique_nodes
        self.embeddings = theta_init
        self.emb_dict = embedding_dict
        
    
    def negative_sample(self, u):
        positives = [x[1] for x in self.relations if x[0] == u]
        negatives = np.array([x for x in self.nodes if x not in positives])
        #negatives = np.array([x[1] for x in data if x[1] not in positives])
        random_ix = np.random.permutation(len(negatives))[:self.n_negative]
        neg_samples = [[u, x] for x in negatives[random_ix]]
        neg_samples.append([u,u])
        return neg_samples
    
    def partial_d(self, theta, x):
        alpha = 1 - norm(theta)**2
        beta = 1 - norm(x)**2
        gamma = 1 + 2/(alpha*beta + self.eps) * norm(theta-x)**2
        lhs = 4 / (beta*np.sqrt(gamma**2 - 1) + self.eps)
        rhs = 1/(alpha**2 + self.eps) * (norm(x)**2 - 2*np.inner(theta,x) + 1) * theta - x/(alpha + self.eps)
        return lhs*rhs
        
    def proj(self, theta):
        if norm(theta) >= 1:
            theta = theta/norm(theta) - self.eps
        return theta
    
    def update(self, u, grad):
        theta = self.emb_dict[u]
        step = 1/4 * self.eta*(1 - norm(theta)**2)**2 * grad
        self.emb_dict[u] = self.proj(theta - step)
        
    def train(self, num_epochs=10):
        for i in range(num_epochs):
            #loss=0
            start = time.time()
            for relation in self.relations:
                u, v = relation[0], relation[1]
                if u == v:
                    continue
                # embedding vectors (theta, x) for relation (u, v)
                theta, x = self.emb_dict[u], self.emb_dict[v]
                # embedding vectors v' in sample negative relations (u, v')
                neg_relations = [x[1] for x in self.negative_sample(u)]
                neg_embed = np.array([self.emb_dict[x] for x in neg_relations])
                # find partial derivatives of poincare distance
                dd_theta = self.partial_d(theta, x) 
                dd_x = self.partial_d(x, theta)
                # find partial derivatives of loss function
                dloss_theta = -1
                dloss_x = 1
                grad_theta = dloss_theta * dd_theta
                grad_x = dloss_x * dd_x
                self.update(u, grad_theta)
                self.update(v, grad_x)
                # find gradients for negative samples
                neg_exp_dist = np.array([np.exp(-poincare_dist(theta, v_prime)) for v_prime in neg_embed])
                Z = neg_exp_dist.sum(axis=0)
                for vprime in neg_relations:
                    dd_vprime = self.partial_d(self.emb_dict[vprime], theta)
                    dd_u = self.partial_d(theta, self.emb_dict[vprime])
                    dloss_vprime = -np.exp(-poincare_dist(self.emb_dict[vprime], theta)) / Z
                    dloss_u = -np.exp(-poincare_dist(theta, self.emb_dict[vprime])) / Z
                    grad_vprime = dd_vprime * dloss_vprime
                    grad_u = dd_u * dloss_u
                    self.update(vprime, grad_vprime)
                    self.update(u, grad_u)
                #loss = loss + np.log(np.exp(-poincare_dist(theta, x))) / Z
            
            print('COMPLETED EPOCH ', i+1)
            #print(' LOSS: ', loss)
            print('---------- total seconds: ', time.time() - start)

mammal = pd.read_csv('mammal_closure.csv')
mammal_relations = [[mammal.id1[i], mammal.id2[i]] for i in range(len(mammal))]

def dist_squared(x, y, axis=None):
    return np.sum((x - y)**2, axis=axis)

def get_subtree(relations, embedding_dict, root_node):
    root_emb = embedding_dict[root_node]
    child_nodes = np.array([embedding_dict[x[0]] for x in relations if x[1] == root_node])
    return child_nodes

def plot_embedding(embedding_dict, label_frac=0.001, plot_frac=0.6, save_fig=False):
    fig = plt.figure(figsize=(14,14))
    plt.grid('off')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.axis('off')
    ax = plt.gca()
    embed_vals = np.array(list(embedding_dict.values()))
    plt.xlim([embed_vals.min(0)[0],embed_vals.max(0)[0]])
    plt.ylim([embed_vals.min(0)[1],embed_vals.max(0)[1]])
    keys = list(embedding_dict.keys())
    min_dist_2 = label_frac * max(embed_vals.max(axis=0) - embed_vals.min(axis=0)) ** 2
    labeled_vals = np.array([2*embed_vals.max(axis=0)])
    groups = [keys[i] for i in np.argsort(np.linalg.norm(embed_vals, axis=1))][:10]
    groups.insert(0, 'mammal.n.01')
    for key in groups:
        if np.min(dist_squared(embedding_dict[key], labeled_vals, axis=1)) < min_dist_2:
            continue
        else:
            _ = ax.scatter(embedding_dict[key][0], embedding_dict[key][1])
            props = dict(boxstyle='round', lw=2, edgecolor='black', alpha=0.5)
            _ = ax.text(embedding_dict[key][0], embedding_dict[key][1]+0.01, s=key.split('.')[0], 
                        size=14, fontsize=16, verticalalignment='top', bbox=props)
            labeled_vals = np.vstack((labeled_vals, embedding_dict[key]))
    n = int(plot_frac*len(embed_vals))
    for i in np.random.permutation(len(embed_vals))[:n]:
        _ = ax.scatter(embed_vals[i][0], embed_vals[i][1])
        if np.min(dist_squared(embed_vals[i], labeled_vals, axis=1)) < min_dist_2:
            continue
        else:
            _ = ax.text(embed_vals[i][0], embed_vals[i][1]+0.02, s=keys[i].split('.')[0], 
                        size=8, fontsize=10, verticalalignment='top', bbox=props)
            labeled_vals = np.vstack((labeled_vals, embed_vals[i]))
    if save_fig:
        plt.savefig('poincare_viz.png')
    print(labeled_vals.shape)


