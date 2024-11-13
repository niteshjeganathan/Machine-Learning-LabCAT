# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Import Dataset 
dataset = pd.read_csv('gait.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Training and Test Data
X_train,  X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
rs = RobustScaler()
X_train_scaled = rs.fit_transform(X_train)
X_test_scaled = rs.transform(X_test)

# Training  Model
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train_scaled, y_train)

# Predicting Results
y_pred = regressor.predict(X_test_scaled)

# Accuracy Metrics
print(mean_absolute_percentage_error(y_test, y_pred))

