# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Importing Dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Label Encoding 
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Comparing Scaling
scalers = {
    "No Scaling": None,
    "Min Max Scaling": MinMaxScaler(), 
    "Standarisation": StandardScaler(), 
    "Robust Scaler": RobustScaler()
}

accuracy_scores = {}

for scaler_name, scaler in scalers.items(): 
    if(scaler):
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy_scores[scaler_name] = accuracy_score(y_test, y_pred)
    else:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy_scores[scaler_name] = accuracy_score(y_test, y_pred)

# Plotting Comparision
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.xlabel("Scaling Technique")
plt.ylabel("Accuracy")
plt.title("Comparision of Scaling Techniques")
plt.show()

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


# Cross Validation 
k_values = 



