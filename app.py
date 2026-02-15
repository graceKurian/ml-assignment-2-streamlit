import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Enterprise Credit Risk Analytics",
    layout="wide"
)

# ---------------------------------------------------
# ELITE DARK UI + STICKY HEADER
# ---------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
    color: #f1f5f9;
}

/* Sticky Header */
.sticky-header {
    position: sticky;
    top: 0;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(6px);
    padding: 15px 0;
    z-index: 999;
    border-bottom: 1px solid #1f2937;
}

/* Header Text */
.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
}

.subtitle {
    color: #94a3b8;
    font-size: 14px;
}

/* Divider */
hr {
    border: 1px solid #1f2937;
    margin: 2rem 0;
}

/* KPI STRIP */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #1a2438);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #334155;
    transition: 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    border-color: #3b82f6;
}

.kpi-label {
    font-size: 13px;
    color: #94a3b8;
}

.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #60a5fa;
}

/* File uploader glow */
.stFileUploader > div {
    background: linear-gradient(135deg, #1e293b, #1a2438);
    border: 2px dashed #3b82f6 !important;
    border-radius: 16px !important;
    padding: 30px !important;
    transition: 0.3s ease;
}

.stFileUploader > div:hover {
    box-shadow: 0 0 25px rgba(59,130,246,0.5);
}

/* Drag text */
.stFileUploader div div span {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* Limit text */
.stFileUploader small {
    font-size: 14px !important;
    color: #94a3b8 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    min-height: 50px !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 3rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# STICKY HEADER
# ---------------------------------------------------
st.markdown("""
<div class="sticky-header">
    <div class="main-title">🏦 Enterprise Credit Risk Analytics</div>
    <div class="subtitle">Advanced machine learning models for credit default prediction</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------
# SAMPLE DATASET
# ---------------------------------------------------
st.header("📥 Sample Dataset")

with open("data/UCI_Credit_Card.csv", "rb") as file:
    st.download_button(
        label="Download Credit Card Dataset",
        data=file,
        file_name="UCI_Credit_Card.csv",
        mime="text/csv"
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------
st.header("⚙️ Model Configuration")

col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost",
        ]
    )

with col2:
    st.markdown("### 📂 Upload Test CSV File")
    uploaded_file = st.file_uploader("", type=["csv"])

# ---------------------------------------------------
# MODEL LOADER
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
# EVALUATION
# ---------------------------------------------------
if uploaded_file is not None:

    st.success("Evaluation Started ✔")

    df = pd.read_csv(uploaded_file)

    y_true = df["default.payment.next.month"]
    X = df.drop(columns=["default.payment.next.month", "ID"], errors="ignore")

    model, scaler = load_model(model_name)

    if scaler is not None:
        X = scaler.transform(X)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # KPI STRIP
    # ---------------------------------------------------
    st.header("📊 Key Performance Indicators")

    kpi_data = [
        ("Accuracy", acc),
        ("Precision", prec),
        ("Recall", rec),
        ("F1 Score", f1),
        ("AUC", auc),
        ("MCC", mcc),
    ]

    cols = st.columns(6)

    for col, (label, value) in zip(cols, kpi_data):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------
    st.header("📈 Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax
    )

    ax.set_facecolor("#0b1220")
    fig.patch.set_facecolor("#0b1220")

    st.pyplot(fig)

st.markdown("<div class='footer'>Enterprise Credit Risk Analytics Platform</div>", unsafe_allow_html=True)
