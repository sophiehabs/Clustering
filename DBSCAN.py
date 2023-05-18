# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:47:44 2023

@author: Pablo
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score


features, true_labels = make_moons(
    n_samples=250, noise=0.05, random_state=42
 )

# scale data

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Instantiate k-means and dbscan algorithms
kmeans = KMeans(n_clusters=2)
dbscan = DBSCAN(eps=0.3)

# Fit the algorithms to the features
kmeans.fit(scaled_features)
dbscan.fit(scaled_features)

# Compute the silhouette scores for each algorithm
sK = kmeans_silhouette = silhouette_score(
    scaled_features, kmeans.labels_
 ).round(2)

sD = dbscan_silhouette = silhouette_score(
 scaled_features, dbscan.labels_
).round (2)


print('Silhouette score for KMeans:')
print(sK)
print('Silhouette score for DBSCAN:')
print(sD)

# A higher silhouette coefficient suggests better clusters, which is misleading in this scenario

# Plot the data and cluster silhouette comparison
fig, (ax1, ax2) = plt.subplots(
     1, 2, figsize=(8, 6), sharex=True, sharey=True
 )
fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
fte_colors = {
     0: "#008fd5",
    1: "#fc4f30",
 }
# The k-means plot
km_colors = [fte_colors[label] for label in kmeans.labels_]
ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
ax1.set_title(
   f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
)

# The dbscan plot
db_colors = [fte_colors[label] for label in dbscan.labels_]
ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
ax2.set_title(
     f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
 )
plt.show()

# evaluating clustering by comparing with true labels

ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)
ari_dbscan = adjusted_rand_score(true_labels, dbscan.labels_)

ari_k = round(ari_kmeans, 2)
print('adjusted rand index (ARI) for KMeans:')
print(ari_k)

ari_d = round(ari_dbscan, 2)
print('adjusted rand index (ARI) for DBSCAN:')
print(ari_d)