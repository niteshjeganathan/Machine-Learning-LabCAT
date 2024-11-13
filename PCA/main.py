# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor

# Importing dataset
dataset =  pd.read_csv('gait.csv')

# Check for Missing Values
# print(dataset.isnull().sum())

# Splitting Target and Features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Data (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)

# PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(X_train_pca)
# Model Training
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test Set Results
y_pred = regressor.predict(X_test)

# Predicting a new variable

# print(regressor.predict([[1,1,1,1,1,100]]))

# Printing Error
# print(mean_squared_error(y_test, y_pred))
# print(mean_absolute_percentage_error(y_test, y_pred))

# Printing Coefficients
# print(regressor.coef_)
# print(regressor.intercept_)

# Updating Dataset
dataset['gait speed'] = 1 * dataset['subject'] + 2 * dataset['condition'] + 3 * dataset['replication'] + 4 * dataset['leg'] + 5 * dataset['joint'] + 6 * dataset['time'] + 7 * dataset['angle']

# Splitting Features and Target
X_new = dataset.iloc[:, :-1].values
y_new = dataset.iloc[:, -1].values

# Splitting Training and Test Data
x_tr, x_te, y_tr, y_te = train_test_split(X_new, y_new)

# Training Model
reg = LinearRegression()
reg.fit(x_tr, y_tr)

# Predict Gait Speed
y_pr = reg.predict(x_te)

# Printing Error
# print(mean_absolute_percentage_error(y_te, y_pr))
# print(reg.coef_)       
# print(reg.intercept_)                                                                                                                                          