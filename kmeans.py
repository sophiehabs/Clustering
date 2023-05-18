# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:01:45 2023

@author: Pablo
"""

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

features, true_labels = make_blobs(
     n_samples=200,
     centers=3,
     cluster_std=2.75,
     random_state=42
 )

print(features[:5])

print(true_labels[:5])

# scale data

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print(scaled_features[:5])

# initialize

kmeans = KMeans(
     init="random",
     n_clusters=3,
     n_init=10,
     max_iter=300,
    random_state=42
 )

# fit

kmeans.fit(scaled_features)

# Statistics from the initialization run with the lowest SSE 
#    are available as attributes of kmeans after calling .fit():
    
# The lowest SSE value
print('The lowest SSE value')
print(kmeans.inertia_)

# Final locations of the centroid
print('Final locations of the centroids')
print(kmeans.cluster_centers_)

# The number of iterations required to converge
print('The number of iterations required to converge')
print(kmeans.n_iter_)

# The cluster assignments
print('The cluster assignments')
print(kmeans.labels_[:5])

#
#    Running the process for various values of K
#

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
     "max_iter": 300,
     "random_state": 42,
 }

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    #  uses Python’s dictionary unpacking operator (**) to iterate through a dictionary
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)
    
# visualize    
    
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# locate elbow

kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
 )
   
print('Elbow is at:')
print(kl.elbow) 


#  using silhouette coefficient as measure of cluster cohesion and separation
# A list holds the silhouette coefficients for each k

silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
     kmeans.fit(scaled_features)
     score = silhouette_score(scaled_features, kmeans.labels_)
     silhouette_coefficients.append(score)

# visualize
     
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

