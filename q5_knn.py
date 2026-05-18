import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("datasets/diabetes1.csv").dropna()
X, y = df.drop("Outcome", axis=1).values, df["Outcome"].values

# Pre-processing: replace zeros with median for physiological columns
zero_cols = [1, 2, 3, 4, 5]  # Glucose, BP, Skin, Insulin, BMI
for c in zero_cols:
    med = np.median(X[X[:, c] != 0, c])
    X[X[:, c] == 0, c] = med

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Custom KNN ---
def euclidean(a, b): return np.sqrt(np.sum((a - b)**2))

def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        dists = [euclidean(x, xt) for xt in X_train]
        k_idx = np.argsort(dists)[:k]
        labels = y_train[k_idx]
        preds.append(np.bincount(labels.astype(int)).argmax())
    return np.array(preds)

y_pred_custom = knn_predict(X_train, y_train, X_test, k=5)
print(f"Custom KNN (k=5) Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")

# --- Sklearn KNN with tuning ---
results = {}
for k in [3, 5, 7, 11, 15]:
    for metric in ["euclidean", "manhattan"]:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[(k, metric)] = acc
        print(f"k={k:2d}, metric={metric:12s}, Accuracy={acc:.4f}")

# Plot k vs accuracy (euclidean)
ks = [3,5,7,11,15]
accs = [results[(k,"euclidean")] for k in ks]
plt.figure(figsize=(6,4))
plt.plot(ks, accs, "bo-"); plt.xlabel("K"); plt.ylabel("Accuracy")
plt.title("KNN: K vs Accuracy (Euclidean)"); plt.grid(True)
plt.tight_layout(); plt.savefig("q5_knn.png", dpi=100); plt.show()
