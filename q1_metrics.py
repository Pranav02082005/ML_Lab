import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, roc_curve, auc)

# Load dataset
df = pd.read_csv("datasets/diabetes1.csv").dropna()
X, y = df.drop("Outcome", axis=1), df["Outcome"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_random = np.random.rand(len(y_test))

# --- Custom metric functions ---
def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    spec = tn / (tn + fp)
    npv = tn / (tn + fn)
    return tp, tn, fp, fn, acc, prec, rec, f1, mcc, spec, npv

tp, tn, fp, fn, acc, prec, rec, f1, mcc, spec, npv = metrics(y_test, y_pred)

print("=== Custom Functions ===")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
print(f"MCC={mcc:.4f}, Specificity={spec:.4f}, NPV={npv:.4f}")

print("\n=== Sklearn Functions ===")
print(f"Accuracy={accuracy_score(y_test,y_pred):.4f}")
print(f"Precision={precision_score(y_test,y_pred):.4f}")
print(f"Recall={recall_score(y_test,y_pred):.4f}")
print(f"F1={f1_score(y_test,y_pred):.4f}")
print(f"MCC={matthews_corrcoef(y_test,y_pred):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test,y_pred)}")

# --- ROC & AUC ---
fpr_m, tpr_m, _ = roc_curve(y_test, y_prob)
fpr_r, tpr_r, _ = roc_curve(y_test, y_random)
auc_m = auc(fpr_m, tpr_m)
auc_r = auc(fpr_r, tpr_r)

print(f"\nAUC (Model)={auc_m:.4f}, AUC (Random)={auc_r:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(fpr_m, tpr_m, label=f"Model AUC={auc_m:.2f}")
plt.plot(fpr_r, tpr_r, label=f"Random AUC={auc_r:.2f}", linestyle="--")
plt.plot([0,1],[0,1],"k:")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.legend(); plt.tight_layout(); plt.savefig("q1_roc.png", dpi=100); plt.show()
