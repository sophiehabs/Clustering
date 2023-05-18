# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:15:57 2019

@author: Pablo
"""

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])


plt.scatter(X[:, 0], X[:, 1], s=50);

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

print(kmeans.labels_)

print(kmeans.predict([[0, 0], [4, 4]]))

print(kmeans.cluster_centers_)

# plot them in colours as distinguished by  kmeans.labels_s, assigned by colour map named 'viridids'
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis')