# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:15:55 2019

@author: Pablo
"""

# import statements
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# create blobs
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)

# create np array for data points
points = data[0]

# create scatter plot
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='viridis')
plt.xlim(-15,15)
plt.ylim(-15,15)

plt.show()

# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

algo = sch.linkage(points, method='ward')

# create dendrogram
dendrogram = sch.dendrogram(algo)


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')


# save clusters for chart
y_hc = hc.fit_predict(points)
