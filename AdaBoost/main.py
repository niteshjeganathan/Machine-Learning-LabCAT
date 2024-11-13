# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor  # For regression tasks
from sklearn.tree import DecisionTreeRegressor  # Base estimator
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Importing dataset
dataset = pd.read_csv('gait.csv')

# Splitting Target and Features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Data (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initializing AdaBoost Regressor with Decision Tree as base estimator
base_estimator = DecisionTreeRegressor(max_depth=3)  # Limiting depth to avoid overfitting
adaboost_regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50)

# Training AdaBoost model
adaboost_regressor.fit(X_train, y_train)

# Predicting Test Set Results
y_pred = adaboost_regressor.predict(X_test)

# Printing Errors
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, y_pred))

# Printing Feature Importances (optional)
print("Feature Importances:", adaboost_regressor.feature_importances_)
