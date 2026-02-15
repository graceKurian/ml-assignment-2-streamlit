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
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Enterprise Credit Risk Analytics",
    layout="wide"
)

# ---------------------------------------------------
# ULTRA PREMIUM DARK THEME
# ---------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: #ffffff;
}

/* Sticky Header */
header {
    visibility: hidden;
}

/* Container spacing */
.block-container {
    padding-top: 2rem;
}

/* Titles */
h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px;
}

h2 {
    margin-top: 2rem !important;
    font-weight: 600 !important;
}

/* Glass panels */
.glass {
    background: rgba(30, 41, 59, 0.55);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 30px;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #334155, transparent);
    margin: 2rem 0;
}

/* SELECTBOX */
.stSelectbox > div > div {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 2px solid #3b82f6 !important;
    min-height: 52px !important;
    font-weight: 600 !important;
}

.stSelectbox svg {
    fill: #60a5fa !important;
    width: 18px !important;
}

/* FILE UPLOADER */
.stFileUploader > div {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 2px dashed #3b82f6 !important;
    border-radius: 14px !important;
    padding: 24px !important;
}

.stFileUploader label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

.stFileUploader button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}

.stFileUploader small {
    color: #38bdf8 !important;
    font-weight: 600 !important;
}

/* Download button */
.stDownloadButton>button {
    background: linear-gradient(90deg, #10b981, #059669);
    color: #ffffff;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
    border: none;
    box-shadow: 0 4px 15px rgba(16,185,129,0.4);
}

/* KPI Cards */
.metric-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #334155;
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-6px);
    border-color: #3b82f6;
    box-shadow: 0 0 20px rgba(59,130,246,0.4);
}

.metric-title {
    font-size: 14px;
    color: #94a3b8;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #60a5fa;
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
# HEADER
# ---------------------------------------------------
st.title("🏦 Enterprise Credit Risk Analytics")
st.markdown(
    "<div style='color:#cbd5e1; font-size:17px;'>Advanced AI Models for Credit Default Risk Prediction</div>",
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------
# SAMPLE DATASET PANEL
# ---------------------------------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("📥 Sample Dataset")

with open("data/UCI_Credit_Card.csv", "rb") as file:
    st.download_button(
        label="Download Credit Card Dataset",
        data=file,
        file_name="UCI_Credit_Card.csv",
        mime="text/csv"
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL CONFIGURATION PANEL
# ---------------------------------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("⚙️ Model Configuration")

col1, col2 = st.columns(2, gap="large")

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
    uploaded_file = st.file_uploader(
        "📂 Upload Test CSV File",
        type=["csv"]
    )

st.markdown("</div>", unsafe_allow_html=True)

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

    df = pd.read_csv(uploaded_file)

    if "default.payment.next.month" not in df.columns:
        st.error("Target column missing.")
    else:
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

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("📊 Performance Metrics")

        metrics = [
            ("Accuracy", acc),
            ("Precision", prec),
            ("Recall", rec),
            ("F1 Score", f1),
            ("AUC", auc),
            ("MCC", mcc),
        ]

        for i in range(0, 6, 3):
            cols = st.columns(3)
            for col, (label, value) in zip(cols, metrics[i:i+3]):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("📈 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax)
        fig.patch.set_facecolor("#020617")
        ax.set_facecolor("#020617")
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>© Enterprise Credit Risk Analytics Platform</div>", unsafe_allow_html=True)
