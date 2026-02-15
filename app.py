import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide",
)

# ---------------------------------------------------
# Professional Dashboard Styling
# ---------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #f4f6f9;
}

/* Main container padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

/* Header */
.main-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-size: 16px;
    color: #5f6368;
    margin-bottom: 2rem;
}

/* Section Titles */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

/* Card container */
.section-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 2rem;
}

/* Metric boxes */
[data-testid="metric-container"] {
    background-color: #f9fafc;
    border-radius: 8px;
    padding: 15px;
    border: 1px solid #e6e9ef;
}

/* Sidebar spacing */
.css-1d391kg {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header Section
# ---------------------------------------------------
st.markdown("<div class='main-title'>💳 Credit Card Default Risk Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Comparative Evaluation of Machine Learning Models for Financial Risk Prediction</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------
# Sidebar - Model Selection
# ---------------------------------------------------
st.sidebar.title("🔎 Model Selection")

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

model_descriptions = {
    "Logistic Regression": "Linear baseline classifier suitable for interpretable financial modeling.",
    "Decision Tree": "Tree-based model capturing non-linear decision boundaries.",
    "KNN": "Distance-based classifier sensitive to feature scaling.",
    "Naive Bayes": "Probabilistic classifier assuming conditional independence of features.",
    "Random Forest": "Ensemble model improving stability and reducing overfitting.",
    "XGBoost": "Gradient boosting model optimized for high predictive performance."
}

st.sidebar.markdown("### Model Information")
st.sidebar.info(model_descriptions[model_name])

# ---------------------------------------------------
# Dataset Download Section
# ---------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📥 Download Sample Dataset</div>", unsafe_allow_html=True)

with open("data/UCI_Credit_Card.csv", "rb") as file:
    st.download_button(
        label="Download Credit Card Dataset",
        data=file,
        file_name="UCI_Credit_Card.csv",
        mime="text/csv"
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Upload Section
# ---------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📂 Upload Test CSV File</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Model Loader
# ---------------------------------------------------
def load_model(name):
    if name == "Logistic Regression":
        return joblib.load("model/logistic_regression.pkl"), joblib.load("model/scaler.pkl")
    if name == "Decision Tree":
        return joblib.load("model/decision_tree.pkl"), None
    if name == "KNN":
        return joblib.load("model/knn.pkl"), joblib.load("model/knn_scaler.pkl")
    if name == "Naive Bayes":
        return joblib.load("model/naive_bayes.pkl"), None
    if name == "Random Forest":
        return joblib.load("model/random_forest.pkl"), None
    if name == "XGBoost":
        return joblib.load("model/xgboost.pkl"), None

# ---------------------------------------------------
# Main Logic
# ---------------------------------------------------
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

        # ---------------------------------------------------
        # Metrics Section
        # ---------------------------------------------------
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Evaluation Metrics</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col1.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")

        col2.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
        col2.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")

        col3.metric("AUC", f"{roc_auc_score(y_true, y_prob):.4f}")
        col3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Confusion Matrix</div>", unsafe_allow_html=True)

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=1,
            linecolor="white",
            square=True,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:14px; color:gray;'>"
    "Credit Risk Prediction Dashboard | Built with Streamlit & Scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
