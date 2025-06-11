# Program do przewidywania ocaleń z Titanica, bazujący na Soft KNN, Fuzzy KNN, Naiwnym Klasyfikatorze Bayesa i Drzewach Decyzyjnych

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Ładowanie danych 
data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df_numeric = data.copy()
df_numeric['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
df_numeric['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df_numeric = df_numeric.drop(columns=['Name', 'Ticket', 'Cabin'])  # opcjonalnie usuń kolumny tekstowe

# Usunięcie wierszy z brakującymi wartościami (lub można je uzupełnić)
df_numeric = df_numeric.dropna()

# Oblicz macierz korelacji
corr_matrix = df_numeric.corr()

# Wizualizacja macierzy korelacji
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Macierz korelacji Titanic")
plt.show()


# Wykres słupkowy – przeżycie wg płci
plt.figure(figsize=(6, 5))
sns.countplot(x='Sex', hue='Survived', data=data, palette='Set2')
plt.title('Przeżywalność względem płci')
plt.xlabel('Płeć')
plt.ylabel('Liczba osób')
plt.legend(title='Przeżył (1) / Zginął (0)')
plt.grid(True)
plt.show()

# Histogram wieku
plt.figure(figsize=(8, 5))
sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Rozkład wieku pasażerów')
plt.xlabel('Wiek')
plt.ylabel('Liczba pasażerów')
plt.grid(True)
plt.show()

# Przygotowanie danych
age_data = data[['Age', 'Survived']].copy()
age_data['Age'] = age_data['Age'].round()
age_grouped = age_data.groupby('Age').agg({'Survived': ['mean', 'count']})
age_grouped.columns = ['SurvivalRate', 'PassengerCount']

# Wykres słupkowy – przeżycie wg klasy
plt.figure(figsize=(6, 5))
sns.countplot(x='Pclass', hue='Survived', data=data, palette='Set1')
plt.title('Przeżywalność względem klasy podróży')
plt.xlabel('Klasa')
plt.ylabel('Liczba osób')
plt.legend(title='Przeżył (1) / Zginął (0)')
plt.grid(True)
plt.show()


# # Preprocessing - Ekstrakcja cech do modeli
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Preprocessing - usuwanie braków
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Preprocessing - zamiana encoderem zmiennych kategorycznych
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])

le_embarked = LabelEncoder()
data['Embarked'] = le_embarked.fit_transform(data['Embarked'])

# Preprocessing - Ustawianie zmiennych
X = data[features]
y = data[target]

# Preprocessing - Standardowa Standaryzacja StandardScalerem 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Preprocessing - Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Soft k-NN ---
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

# --- Naiwny Bayes ---
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-6
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.vars[c]))
                likelihood -= 0.5 * np.sum(((x - self.means[c]) ** 2) / self.vars[c])
                posteriors.append(prior + likelihood)
            preds.append(self.classes[np.argmax(posteriors)])
        return preds

# --- Drzewo Decyzyjne ---
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _best_split(self, X, y):
        best_gain = -1
        best_split = None
        parent_entropy = self._entropy(y)
        for feat in range(self.n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left = y[X[:, feat] <= t]
                right = y[X[:, feat] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_entropy - (len(left) / len(y)) * self._entropy(left) - (len(right) / len(y)) * self._entropy(right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat, t)
        return best_split

    def _grow_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or depth == self.max_depth:
            return int(np.bincount(y).argmax())
        split = self._best_split(X, y)
        if not split:
            return int(np.bincount(y).argmax())
        feat, t = split
        left_idx = X[:, feat] <= t
        right_idx = X[:, feat] > t
        left_tree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return (feat, t, left_tree, right_tree)

    def _predict_one(self, x, node):
        if isinstance(node, int):
            return node
        feat, t, left, right = node
        if x[feat] <= t:
            return self._predict_one(x, left)
        else:
            return self._predict_one(x, right)

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

# --- Fuzzy k-NN ---
def fuzzy_vote_knn(X_train, y_train, X_test, k):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    predictions = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        indices = np.argsort(dists)[:k]
        votes = {}
        for i in indices:
            label = y_train[i]
            mu = 1 / (1 + dists[i])
            votes[label] = votes.get(label, 0) + mu
        pred = max(votes.items(), key=lambda x: x[1])[0]
        predictions.append(pred)
    return predictions

# Inicjalizacja
k_values = range(1, 16, 2)
soft_knn_results = []
fuzzy_knn_results = []

for k in k_values:
    soft_knn = SoftKNN(k=k)
    soft_knn.fit(X_train, y_train)
    y_pred_soft = soft_knn.predict(X_test)
    acc_soft = accuracy_score(y_test, y_pred_soft)
    soft_knn_results.append((k, acc_soft))

    y_pred_fuzzy = fuzzy_vote_knn(X_train, y_train, X_test, k=k)
    acc_fuzzy = accuracy_score(y_test, y_pred_fuzzy)
    fuzzy_knn_results.append((k, acc_fuzzy))

best_soft_k, best_soft_acc = max(soft_knn_results, key=lambda x: x[1])
best_fuzzy_k, best_fuzzy_acc = max(fuzzy_knn_results, key=lambda x: x[1])

print(f"\nBest Soft k-NN Accuracy: {best_soft_acc:.4f} with k = {best_soft_k}")
print(f"Best Fuzzy k-NN Accuracy: {best_fuzzy_acc:.4f} with k = {best_fuzzy_k}")

# Trenowanie modeli dla k=5
soft_knn = SoftKNN(k=best_soft_k)
soft_knn.fit(X_train, y_train)
y_pred_soft = soft_knn.predict(X_test)

nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

tree = DecisionTree(max_depth=4)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

y_pred_fuzzy = fuzzy_vote_knn(X_train, y_train, X_test, k=best_fuzzy_k)

# Ewaluacja
models = {
    "Soft k-NN": y_pred_soft,
    "Naive Bayes": y_pred_nb,
    "Decision Tree": y_pred_tree,
    "Fuzzy k-NN": y_pred_fuzzy
}

for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(classification_report(y_test, preds))


for name, preds in models.items():
    ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=["Zginął", "Przeżył"],
                                            cmap="Blues")
    plt.title(f'Macierzy pomyłek – {name}')
    plt.grid(False)
    plt.show()

# Wyświetlanie rezulatu końcowego - wykres + raport
plt.figure(figsize=(10, 5))
plt.plot([k for k, _ in soft_knn_results], [acc for _, acc in soft_knn_results], label='Soft k-NN')
plt.plot([k for k, _ in fuzzy_knn_results], [acc for _, acc in fuzzy_knn_results], label='Fuzzy k-NN')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k for Soft k-NN and Fuzzy k-NN')
plt.legend()
plt.grid(True)
plt.show()


