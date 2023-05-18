# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:27:15 2023

@author: Pablo
"""

#Importing required libraries
 
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
 
 
#Getting the data ready
 
data = load_iris()
df = data.data
#Selecting certain features based on which clustering is done 
df = df[:,1:3]


#Linkage Matrix
from scipy.cluster.hierarchy import dendrogram , linkage

Z = linkage(df, method = 'ward')
 
#plotting dendrogram
dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()
 
 
#Creating the model
 
agg_clustering = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
 
#predicting the labels
 
labels = agg_clustering.fit_predict(df)
 
#Plotting the results
 
plt.figure(figsize = (8,5))
plt.scatter(df[labels == 0 , 0] , df[labels == 0 , 1] , c = 'red')
plt.scatter(df[labels == 1 , 0] , df[labels == 1 , 1] , c = 'blue')
plt.scatter(df[labels == 2 , 0] , df[labels == 2 , 1] , c = 'green')
plt.show()