# Importing Libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Importing Dataset
dataset = pd.read_csv('adult.data', header=None)

# Missing Values
dataset.replace([' ?'], np.nan, inplace=True)
print(dataset.isnull().sum())

for col in dataset.columns: 
    mode = dataset[col].mode()[0]
    dataset[col].fillna(mode, inplace=True)

print(dataset.isnull().sum())

# Splitting Features and Target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Encoding Features
categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]
les = []

for col in categorical_columns:
    le = LabelEncoder()
    X_train[:, col] = le.fit_transform(X_train[:, col])
    X_test[:, col] = le.transform(X_test[:, col])

    les.append(le)

# Encoding Target 
le_target = LabelEncoder()
y_train = le_target.fit_transform(y_train)
y_test = le_target.transform(y_test)

# Feature Scaling
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.fit_transform(X_test)

# Training Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predicting Test Set Results
y_pred = model.predict(X_test_scaled)

# Accuracy Metrics
print(classification_report(y_test, y_pred))