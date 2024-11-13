# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Importing Dataset
dataset = pd.read_csv('iris.data')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label Encoding 
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

# Model Training
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting Test Set Results
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=classifier.classes_)
disp.plot()
plt.show()
