# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

# Importing Dataset 
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label Encoding Target Values
le = LabelEncoder()
y = le.fit_transform(y)

# Defining Scalers
scalers = {
    "No Scaling" : None, 
    "Min Max Scaling": MinMaxScaler(), 
    "Standard Scaling": StandardScaler(), 
    "Robust Scaling": RobustScaler()
}

# Splitting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Comparing Performance for each scaler
accuracy_scores = {}
for scaler_name, scaler in scalers.items(): 
    if(scaler):
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
        classifier.fit(X_train_scaled, y_train)

        y_pred = classifier.predict(X_test_scaled)
        accuracy_scores[scaler_name] = accuracy_score(y_test, y_pred)
    else:
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy_scores[scaler_name] = accuracy_score(y_test, y_pred)

# Plotting the performance 
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.show()

# Cross Validation and Plotting the scores of accuracy scores against k
cross_scores = []
k_range = range(1, 21)
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='accuracy')
    cross_scores.append(scores.mean())

plt.plot(k_range, cross_scores)
plt.show()

# ROC Curves
y_test_binarised = label_binarize(y_test, classes=classifier.classes_)
n_classes = y_test_binarised.shape[1]

classifier_ovr = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
y_score = classifier_ovr.fit(X_train_scaled, y_train).predict_proba(X_test_scaled)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarised[:, i], y_score[:, i])
    roc_auc[i] = roc_auc_score(y_test_binarised[:, i], y_score[:, i])

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class: {classifier.classes_[i]}, AUC: {roc_auc[i]}')

plt.legend()
plt.show()