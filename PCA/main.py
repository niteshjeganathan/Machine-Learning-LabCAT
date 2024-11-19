# Importing Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Importing Dataset 
dataset = pd.read_csv('iris.data')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label Encoding Target Values
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# Comparing Performance with and without PCA
methods = {
    "PCA": PCA(n_components=3), 
    "No PCA": None
}

accuracy_scores = {}

for method_name, method in methods.items():
    if(method):
        X_train_transformed = method.fit_transform(X_train_scaled)
        X_test_transformed = method.transform(X_test_scaled)
    else:
        X_train_transformed = X_train_scaled
        X_test_transformed = X_test_scaled
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train_transformed, y_train)

    y_pred = classifier.predict(X_test_transformed)

    score = accuracy_score(y_test, y_pred)

    accuracy_scores[method_name] = score

# Plotting Comparision
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.show()