import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

# ---- LINEAR REGRESSION (Advertising: TV -> sales) ----
df_adv = pd.read_csv("datasets/Advertising.csv", index_col=0)
X_lr = df_adv[["TV"]]
y_lr = df_adv["sales"]
X_tr, X_te, y_tr, y_te = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_tr, y_tr)
y_pred_lr = lr.predict(X_te)
mse = mean_squared_error(y_te, y_pred_lr)
print("=== Linear Regression (TV -> Sales) ===")
print(f"Coef={lr.coef_[0]:.4f}, Intercept={lr.intercept_:.4f}")
print(f"MSE={mse:.4f}, RMSE={np.sqrt(mse):.4f}")
print(f"New prediction for TV=150: {lr.predict([[150]])[0]:.2f}")

plt.figure(figsize=(6,4))
plt.scatter(X_te, y_te, color="blue", label="Actual")
plt.plot(X_te, y_pred_lr, color="red", label="Predicted")
plt.xlabel("TV"); plt.ylabel("Sales"); plt.title("Linear Regression")
plt.legend(); plt.tight_layout(); plt.savefig("q3_linear.png", dpi=100); plt.show()

# ---- LOGISTIC REGRESSION (Diabetes) ----
df_d = pd.read_csv("datasets/diabetes1.csv").dropna()
X_log = df_d.drop("Outcome", axis=1)
y_log = df_d["Outcome"].astype(int)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000).fit(X_tr2, y_tr2)
y_pred_log = log_reg.predict(X_te2)
print("\n=== Logistic Regression (Diabetes) ===")
print(f"Accuracy={accuracy_score(y_te2, y_pred_log):.4f}")
print(f"Precision={precision_score(y_te2, y_pred_log):.4f}")
print(f"Recall={recall_score(y_te2, y_pred_log):.4f}")
print(f"F1={f1_score(y_te2, y_pred_log):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_te2, y_pred_log)}")
new_sample = [[2, 120, 70, 20, 79, 28.0, 0.5, 33]]
print(f"New sample prediction: {'Diabetic' if log_reg.predict(new_sample)[0]==1 else 'Not Diabetic'}")
