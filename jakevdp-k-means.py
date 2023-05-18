# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:21:38 2019

@author: Pablo
"""


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

# generate random set of points

from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
# plot them 

plt.scatter(X[:, 0], X[:, 1], s=50);

# run k-means on the data
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# plot them in colours as distinguished by  y_kmeans, assigned by colour map named 'viridids'
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

#  plot the centroids as they stood at end of computation
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);