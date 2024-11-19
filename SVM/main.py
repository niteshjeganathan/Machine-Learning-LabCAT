# Importing Libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hinge_loss

# Importing Dataset 
dataset = pd.read_csv('breast-cancer.data')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label Encoding Features
les = []
for i in range(X.shape[1]):
    le = LabelEncoder()
    X[:, i] = le.fit_transform(X[:, i])
    les.append(le)

# Label Encoding Target
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Splitting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

# Setting up Linear and Non Linear
models = {
    "Linear" : SVC(kernel='linear'), 
    "Non Linear": SVC(kernel='rbf')
}

# Comparing Performance Measures
accuracy_scores = {}
hinge_losses = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    accuracy_scores[model_name] = score
    
    decision_values = model.decision_function(X_test)
    hinge_loss_value = hinge_loss(y_test, decision_values)
    hinge_losses[model_name] = hinge_loss_value

plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.show()

plt.bar(hinge_losses.keys(), hinge_losses.values())
plt.show()