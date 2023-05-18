# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:26:49 2023

@author: Pablo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv('Wholesale customers data.csv')
print(data.head())

# normalise

from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
print(data_scaled.head())

# dendrogram

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

# with line

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=6, color='r', linestyle='--')


# predict

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
prediction = cluster.fit_predict(data_scaled)


print(prediction)

# visualise

plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_) 