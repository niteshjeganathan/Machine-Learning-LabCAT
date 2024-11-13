# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Importing dataset
dataset = pd.read_csv('gait.csv')

# Splitting Target and Features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Data (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initializing XGBoost Regressor
xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)

# Training the XGBoost Regressor
xgb_regressor.fit(X_train, y_train)

# Predicting Test Set Results
y_pred = xgb_regressor.predict(X_test)

# Printing Errors
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, y_pred))

# Feature Importance
xgb_regressor.feature_importances_
print("Feature Importances:", xgb_regressor.feature_importances_)
