# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

# Importing Dataset
dataset = pd.read_csv("dataset.csv")

# Checking Missing values
print(dataset.isnull().sum())
# dataset.fillna(dataset.mean(), inplace=True) # Fills in missing values with  the corresponding mean values

# Splitting Features and Target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding Target Variable
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling : Min-Max Scaling
mm = MinMaxScaler()
X_train_scaled = mm.fit_transform(X_train)
X_test_scaled = mm.transform(X_test)

# Training Model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_scaled, y_train)

# Predicting Test Set Results
y_pred = classifier.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()
