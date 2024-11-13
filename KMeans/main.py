# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Importing Dataset 
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 1:].values

# Choosing k
wcss = []
silhoutte_scores = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

    if(i != 1):
        cluster_labels = kmeans.labels_
        silhoutte = silhouette_score(X, cluster_labels)
        silhoutte_scores.append(silhoutte)

# Plotting Graphs

plt.plot(range(1, 11), wcss)
plt.show()

plt.plot(range(2, 11), silhoutte_scores)
plt.show()

# Selecting k = 2
model = KMeans(n_clusters=2, init='k-means++', random_state=0)
y = model.fit_predict(X)

# Plotting Graphs 
plt.scatter(X[y == 0, 0], X[y==0, 1], s = 100, c='blue', label='Cluster 1')
plt.scatter(X[y == 1, 0], X[y==1, 1], s = 100, c='red', label='Cluster 2')
plt.legend()
plt.show()






