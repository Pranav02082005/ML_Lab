import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,LabelEncoder

df = pd.read_csv("datasets/iris1.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Standardization
X = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Captured:",
      pca.explained_variance_ratio_.sum())

# Visualization
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=LabelEncoder().fit_transform(y)
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")

plt.show()