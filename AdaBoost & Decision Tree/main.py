# Importing Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Importing Dataset
dataset = pd.read_csv('breast-cancer.data')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label Encoding for Features
les = []
for i in range(X.shape[1]): 
    le = LabelEncoder()
    X[:, i] = le.fit_transform(X[:, i])
    les.append(le)

# Label Encoding for Target
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Splitting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Setting up Models 
models = {
    "Random Forest": RandomForestClassifier(), 
    "Ada Boost": AdaBoostClassifier()
}

# Comparing Performance
accuracy_scores = {}

for model_name, model in models.items(): 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy_scores[model_name] = accuracy_score(y_test, y_pred)

# Plotting Graphs
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.show()