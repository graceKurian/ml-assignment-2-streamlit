import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
print(">>> Decision Tree script started")
# 1. Load dataset
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

# 5. Train Decision Tree
model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

print("Decision Tree Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# 8. Save model
joblib.dump(model, "model/decision_tree.pkl")

print("\nDecision Tree model saved successfully.")
