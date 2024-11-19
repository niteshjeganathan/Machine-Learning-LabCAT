# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Importing Dataset
dataset = pd.read_csv('gait.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# Model Training
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

# Predicting Test Values
y_pred = regressor.predict(X_test_scaled)

# Performance Metrics
print("MSE: ", mean_squared_error(y_test, y_pred))
print("R Squared: ", r2_score(y_test, y_pred))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

# Printing Coefficients
print("Coefficients: ", regressor.coef_)
print("Intercept: ", regressor.intercept_) 