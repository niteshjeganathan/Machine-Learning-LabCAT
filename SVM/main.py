# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, hinge_loss

# Importing Dataset  
dataset = pd.read_csv('iris.data', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Split Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.3, random_state=0)

# Feature Scaling
mm =  MinMaxScaler()
X_train_scaled = mm.fit_transform(X_train)
X_test_scaled = mm.transform(X_test)

# Model Training 
classifier =  SVC(kernel='linear')
classifier.fit(X_train_scaled, y_train)

# Predict Results
y_pred = classifier.predict(X_test_scaled)

# Accuracy Metrics
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()


# Hinge Loss 
decision_values = classifier.decision_function(X_test_scaled)
hinge_loss = hinge_loss(y_test, decision_values)
print(hinge_loss)