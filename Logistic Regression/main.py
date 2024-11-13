# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Importing Dataset 
dataset = pd.read_csv('adult.data', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Checking missing values
dataset.replace([' ?', '', '!'], np.nan, inplace=True)
print(dataset.isnull().sum())

for col in dataset.columns:
    mode_value = dataset[col].mode()[0]
    dataset[col].fillna(mode_value, inplace=True)

print(dataset.isnull().sum())

# Label Encoding
categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]
les = []

for col in categorical_columns:
    le = LabelEncoder()
    X[:, col] = le.fit_transform(X[:, col])
    les.append(le)

targetLe = LabelBinarizer()
y = targetLe.fit_transform(y)

# Splitting Training Data and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# Training Model
regressor = LogisticRegression()
regressor.fit(X_train_scaled, y_train)

# Predicting Model
y_pred = regressor.predict(X_test_scaled)

# Printing Metrics
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()


# ROC Curve and AUC for Binary Classification
y_score = regressor.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)

# Plotting ROC Curve
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.show()  