# !pip3 install kmodes
# Importing Librariess
import numpy as np
import pandas  as pd
import matplotlib.pyplot  as  plt
from kmodes.kmodes import KModes

# Importing Dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.values

# Finding K

wcss = []

for i in range(1, 9):  
    model = KModes(n_clusters=i, init='Huang', random_state=0)
    model.fit(X)
    wcss.append(model.cost_)

# Plot 
plt.plot(range(1, 9), wcss)
plt.show()

# Training Model
model = KModes(n_clusters=2, init='Huang', random_state=0)
y = model.fit_predict(X)

# Plot 
plt.scatter(X[y == 0, 0], X[y == 0, 1], s = 100, c = 'blue', label = "Cluster 1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s = 100, c = 'red', label = "Cluster 2")
plt.show()