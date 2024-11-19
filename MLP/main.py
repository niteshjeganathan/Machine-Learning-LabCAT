# Importing Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Importing Dataset
dataset = pd.read_csv('gait.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

# Splitting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling 
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# Setting up Models
models = {
    "SLP" : MLPRegressor(hidden_layer_sizes=(), activation='logistic', max_iter=1000), 
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), activation='logistic', max_iter=1000)
}

# Comparing Performance 
scores = {}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)

    scores[model_name] = mean_absolute_percentage_error(y_test, y_pred)

# Plotting Comparision
plt.bar(scores.keys(), scores.values())
plt.show()


