# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:29:29 2019

@author: Pablo
"""

import matplotlib.pyplot as plt

import numpy as np


# generate random set of points

from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
# plot them 

plt.scatter(X[:, 0], X[:, 1], s=50);

# implement search procedure

from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels


# run k-means on the data
    
centers, labels = find_clusters(X, 4, rseed=6)


# plot them in colours as distinguished by  labels, assigned by colour map named 'viridids'
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');
            

#  plot the centroids as they stood at end of computation
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);