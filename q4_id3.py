import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("datasets/Play_Tennis.csv")
print("Dataset:\n", df.to_string(index=False))

# Encode categorical features
le = LabelEncoder()
df_enc = df.apply(le.fit_transform)
X = df_enc.drop("play", axis=1)
y = df_enc["play"]

# ID3 uses entropy (criterion='entropy')
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)
print(f"\nTraining Accuracy: {accuracy_score(y, model.predict(X)):.4f}")

# Predict new sample: outlook=sunny, temp=cool, humidity=high, windy=false
new = pd.DataFrame([[2,0,0,0]], columns=X.columns)
pred = model.predict(new)[0]
print(f"New sample prediction: {'Play' if pred==1 else 'No Play'}")

# Plot tree
plt.figure(figsize=(10, 5))
plot_tree(model, feature_names=X.columns, class_names=["No","Yes"],
          filled=True, rounded=True, fontsize=10)
plt.title("ID3 Decision Tree - Play Tennis")
plt.tight_layout(); plt.savefig("q4_tree.png", dpi=100); plt.show()
