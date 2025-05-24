# Titanic survival prediction using soft k-NN with fuzzy voting and automatic k selection

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])

le_embarked = LabelEncoder()
data['Embarked'] = le_embarked.fit_transform(data['Embarked'])

# Prepare dataset
X = data[features]
y = data[target]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Automatic k selection for k-NN ---
k_values = list(range(1, 21))
k_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

best_k = k_values[np.argmax(k_scores)]
print(f"Best k based on cross-validation: {best_k}")

# --- Soft k-NN using fuzzy similarity for categorical fields (Sex, Embarked) ---
class SoftKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)


    def predict(self, X):
        predictions = []
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = np.linalg.norm(x - x_train)
                distances.append((dist, self.y_train[i]))
            distances.sort(key=lambda tup: tup[0])
            neighbors = distances[:self.k]
            weights = [1 / (d + 1e-5) for d, _ in neighbors]
            votes = {}
            for (_, label), weight in zip(neighbors, weights):
                votes[label] = votes.get(label, 0) + weight
            pred = max(votes.items(), key=lambda x: x[1])[0]
            predictions.append(pred)
        return predictions

# Train and evaluate soft k-NN
soft_knn = SoftKNN(best_k)
soft_knn.fit(X_train, y_train)
y_pred_soft = soft_knn.predict(X_test)

print("\nSoft k-NN Results:")
print(classification_report(y_test, y_pred_soft))

# --- Classical models for comparison ---
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'k-NN (classic)': KNeighborsClassifier(n_neighbors=best_k)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

# --- Plotting accuracy for different k values ---
plt.plot(k_values, k_scores, marker='o')
plt.xlabel('k')
plt.ylabel('Cross-validated Accuracy')
plt.title('k-NN Accuracy for different k values')
plt.grid()
plt.show()
