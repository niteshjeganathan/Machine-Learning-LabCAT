# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.multiclass import OneVsRestClassifier

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

# Training Model using different scalers
scalers = {
    "No Scaling" : None, 
    "Min-Max Scaling" : MinMaxScaler(), 
    "Standardisation" : StandardScaler(), 
    "Robust Scaling" : RobustScaler()
}

accuracy_scores = {}

for scaling_name, scaler in scalers.items(): 
    if(scaler):
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train_scaled, y_train)

        y_pred = classifier.predict(X_test_scaled)
        accuracy_scores[scaling_name] = accuracy_score(y_test, y_pred)
    else: 
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy_scores[scaling_name] = accuracy_score(y_test, y_pred)
        

# Plotting Accuracy Scores
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.xlabel("Scaling Technique")
plt.ylabel("Accuracy")
plt.show()
print(accuracy_scores)

# Choosing Robust Scaling

# Scaling
rs = RobustScaler()
X_train_scaled = rs.fit_transform(X_train)
X_test_scaled = rs.transform(X_test)

# Training Model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_scaled, y_train)

# Predicting Test Results
y_pred = classifier.predict(X_test_scaled)

# Predicting Individual Result
print(le.inverse_transform(classifier.predict(rs.transform([[42020, 674.16, 208.81, 162.14, 1.29, 0.5174, 42530, 
                                        231.01, 0.7213, 0.9880, 0.92, 0.91, 0.4949, 0.9637, 0.9975, 0.9128]]))))


# Cross Validation and Plotting Accuracy vs K
k_range = range(1, 21)
cv_scores = []

for k in k_range: 
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv = 10, scoring='accuracy')
    cv_scores.append(scores.mean())

plt.plot(k_range, cv_scores)
plt.xlabel('K Value')
plt.ylabel('CV Score')
plt.show()

# Accuracy Metrics
print(classification_report(y_test, y_pred))

# ROC Curves and AUC Curves
y_test_binarised = label_binarize(y_test, classes=classifier.classes_)
n_classes = y_test_binarised.shape[1]

classifier_ovr = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
y_score = classifier_ovr.fit(X_train, y_train).predict_proba(X_test)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes): 
    fpr[i], tpr[i], _ = roc_curve(y_test_binarised[:, i], y_score[:, i])
    roc_auc[i] = roc_auc_score(y_test_binarised[:, i], y_score[:, i])

# Plotting ROC Curves

for i in range(n_classes): 
    plt.plot(fpr[i], tpr[i], label=f'Class {classifier.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot()
plt.show()
