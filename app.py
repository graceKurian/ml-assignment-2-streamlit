import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Assignment 2 – Classification Models")

st.write("Upload a CSV file and evaluate different trained models.")

# -------------------------------
# Model Loader
# -------------------------------
def load_model(model_name):
    if model_name == "Logistic Regression":
        model = joblib.load("model/logistic_regression.pkl")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler
    elif model_name == "Decision Tree":
        return joblib.load("model/decision_tree.pkl"), None
    elif model_name == "KNN":
        model = joblib.load("model/knn.pkl")
        scaler = joblib.load("model/knn_scaler.pkl")
        return model, scaler
    elif model_name == "Naive Bayes":
        return joblib.load("model/naive_bayes.pkl"), None
    elif model_name == "Random Forest":
        return joblib.load("model/random_forest.pkl"), None
    elif model_name == "XGBoost":
        return joblib.load("model/xgboost.pkl"), None

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ]
)

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "default.payment.next.month" not in df.columns:
        st.error("Target column 'default.payment.next.month' not found.")
    else:
        y_true = df["default.payment.next.month"]
        X = df.drop(columns=["default.payment.next.month", "ID"], errors="ignore")

        model, scaler = load_model(model_name)

        if scaler is not None:
            X = scaler.transform(X)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Metrics
        st.subheader("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col1.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")

        col2.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
        col2.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")

        col3.metric("AUC", f"{roc_auc_score(y_true, y_prob):.4f}")
        col3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
