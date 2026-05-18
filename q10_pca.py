import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("datasets/diabetes1.csv").dropna()
X = StandardScaler().fit_transform(df.drop("Outcome", axis=1))
y = df["Outcome"].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Captured: {pca.explained_variance_ratio_.sum():.4f}")

plt.figure(figsize=(7, 5))
for label, color, name in [(0,"blue","Non-Diabetic"), (1,"red","Diabetic")]:
    mask = y == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, alpha=0.6, s=30)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA - Diabetes Dataset (2 Components)")
plt.legend(); plt.tight_layout(); plt.savefig("q10_pca.png", dpi=100); plt.show()
