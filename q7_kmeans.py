import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("iris1.csv")

X = df.iloc[:, :2]
wcss = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_
        for k in range(1, 11)]

plt.plot(range(1, 11), wcss, marker="o")
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(12,4))

for i, k in enumerate([2,3,4]):

    model = KMeans(n_clusters=k, random_state=42)

    clusters = model.fit_predict(X)

    axes[i].scatter(
        X.iloc[:,0],
        X.iloc[:,1],
        c=clusters
    )

    axes[i].scatter(
        model.cluster_centers_[:,0],
        model.cluster_centers_[:,1],
        c="black",
        marker="X",
        s=100
    )

    axes[i].set_title(f"K = {k}")

plt.tight_layout()
plt.show()