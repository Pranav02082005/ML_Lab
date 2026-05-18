import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create dataset: weather conditions -> play cricket
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "Temperature": np.random.choice(["Hot","Mild","Cool"], n),
    "Humidity":    np.random.choice(["High","Normal"], n),
    "Outlook":     np.random.choice(["Sunny","Overcast","Rainy"], n),
    "Wind":        np.random.choice(["Strong","Weak"], n),
})
# Rule-based label generation
df["Play"] = ((df["Outlook"]=="Overcast") |
              ((df["Humidity"]=="Normal") & (df["Wind"]=="Weak"))).astype(int)

# Encode
df_enc = pd.get_dummies(df.drop("Play",axis=1))
X, y = df_enc.values, df["Play"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred, target_names=["No Play","Play"]))

# New test prediction
new = pd.get_dummies(pd.DataFrame([{"Temperature":"Mild","Humidity":"Normal",
                                     "Outlook":"Sunny","Wind":"Weak"}]))
new = new.reindex(columns=df_enc.columns, fill_value=0)
print("New sample prediction:", "Play" if model.predict(new)[0]==1 else "No Play")
