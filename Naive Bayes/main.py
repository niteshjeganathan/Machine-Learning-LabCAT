# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score

# Importing Dataset 
dataset = pd.read_csv('iris.data')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label Encoding Target Values
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# Training Model
classifier = GaussianNB()
classifier.fit(X_train_scaled, y_train)

# Predicting Test Results
y_pred = classifier.predict(X_test_scaled)

# Performance Metrics
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print(classification_report(y_test, y_pred))
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
