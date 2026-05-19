import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("datasets/iris1.csv")
X = StandardScaler().fit_transform(df[df.columns[:4]])

# ---- AGNES (Agglomerative - bottom-up) ----
agnes = AgglomerativeClustering(n_clusters=3, linkage="ward")
agnes_labels = agnes.fit_predict(X)

# ---- DIANA (Divisive - top-down, custom) ----
def diana(X, n_clusters):
    clusters = [list(range(len(X)))]
    dist_matrix = squareform(pdist(X))
    
    while len(clusters) < n_clusters:
        # Split the largest cluster
        largest = max(clusters, key=len)
        if len(largest) < 2:
            break
        clusters.remove(largest)
        
        # Find most dissimilar point (average distance to others in cluster)
        avg_dists = [np.mean([dist_matrix[i][j] for j in largest if j != i]) for i in largest]
        splinter_idx = largest[np.argmax(avg_dists)]
        
        group1, group2 = [splinter_idx], [p for p in largest if p != splinter_idx]
        # Reassign based on closer group
        changed = True
        while changed:
            changed = False
            for p in group2[:]:
                d1 = np.mean([dist_matrix[p][q] for q in group1])
                d2 = np.mean([dist_matrix[p][q] for q in group2 if q != p]) if len(group2) > 1 else np.inf
                if d1 < d2:
                    group1.append(p); group2.remove(p); changed = True
        clusters.extend([group1, group2])
    
    labels = np.zeros(len(X), dtype=int)
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = i
    return labels

diana_labels = diana(X, n_clusters=3)

# ---- Plots ----
linkage_matrix = linkage(X, method = "ward")
dendrogram(linkage_matrix, no_labels=True)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

plt.scatter(
    X[:,0],
    X[:,1],
    c=agnes_labels
)
plt.title("Agnes Clustering")
plt.show()

plt.scatter(
    X[:,0],
    X[:,1],
    c=diana_labels
)
plt.title("DIANA Clustering")
plt.show()

print("AGNES Cluster Labels:")
print(agnes_labels)

print("\nDIANA Cluster Labels:")
print(diana_labels)