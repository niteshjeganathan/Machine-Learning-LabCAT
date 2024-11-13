# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('gait.csv')

# Splitting Target and Features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Data (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Single-Layer Perceptron (No Hidden Layers)

single_layer_perceptron = MLPRegressor(hidden_layer_sizes=(), max_iter=1000, activation='logistic')

# Training Single-Layer Perceptron
single_layer_perceptron.fit(X_train, y_train)

# Predicting Test Set Results for Single-Layer Perceptron
y_pred_single = single_layer_perceptron.predict(X_test)

# Evaluating Single-Layer Perceptron
print("Single-Layer Perceptron Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_single))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, y_pred_single))

# Multi-Layer Perceptron (Two hidden layers with 64 and 32 neurons)
multi_layer_perceptron = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=0)

# Training Multi-Layer Perceptron
multi_layer_perceptron.fit(X_train, y_train)

# Predicting Test Set Results for Multi-Layer Perceptron
y_pred_multi = multi_layer_perceptron.predict(X_test)

# Evaluating Multi-Layer Perceptron
print("\nMulti-Layer Perceptron Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_multi))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, y_pred_multi))

# Optional: Plotting Predictions vs Actual for Comparison
plt.figure(figsize=(12, 6))

# Plot for Single-Layer Perceptron
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_single, color="blue", alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Single-Layer Perceptron (No Hidden Layers)")

# Plot for Multi-Layer Perceptron
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_multi, color="green", alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Multi-Layer Perceptron (Hidden Layers)")

plt.tight_layout()
plt.show()
