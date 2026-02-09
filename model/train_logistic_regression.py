import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Load dataset (UPDATE PATH if needed)
df = pd.read_csv("UCI_Credit_Card.csv")

# 2. Drop ID column
df.drop(columns=["ID"], inplace=True)

# 3. Separate features and target
X = df.drop(columns=["default.payment.next.month"])
y = df["default.payment.next.month"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 8. Evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

print("Logistic Regression Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# 9. Save model and scaler
joblib.dump(model, "model/logistic_regression.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and scaler saved successfully.")
