import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/iris1.csv")
X = df[df.columns[:4]]
X_scaled = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = ["red","green","blue","orange"]

for idx, k in enumerate([2, 3, 4]):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    
    ax = axes[idx]
    for c in range(k):
        mask = labels == c
        ax.scatter(X.iloc[mask, 0], X.iloc[mask, 1], c=colors[c], label=f"C{c+1}", s=30)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c="black", marker="X", s=150, label="Centers")
    ax.set_title(f"K={k} | Inertia={inertia:.1f}")
    ax.set_xlabel("Sepal Length"); ax.set_ylabel("Petal Length")
    ax.legend()

plt.suptitle("K-Means Clustering on Iris Dataset")
plt.tight_layout(); plt.savefig("q7_kmeans.png", dpi=100); plt.show()
