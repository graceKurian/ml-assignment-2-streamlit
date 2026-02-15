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

/* App Background */
.stApp {
    background: linear-gradient(180deg, #050a18 0%, #0b1220 100%);
    color: #ffffff;
}

/* Reduce padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headers */
h1 {
    font-size: 40px !important;
    font-weight: 800;
    color: #ffffff !important;
}

h2 {
    font-weight: 700;
    color: #f1f5f9 !important;
    margin-top: 2rem !important;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 1.5rem;
}

/* Divider */
hr {
    border: 1px solid #1e293b;
    margin: 1.8rem 0;
}

/* SELECTBOX */
.stSelectbox > div > div {
    background-color: #111827 !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 12px !important;
    min-height: 55px !important;
    padding-left: 14px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

.stSelectbox svg {
    fill: #60a5fa !important;
}

/* FILE UPLOADER MAIN BOX */
.stFileUploader > div {
    background: rgba(17, 24, 39, 0.95) !important;
    border: 2px dashed #3b82f6 !important;
    border-radius: 16px !important;
    padding: 28px !important;
    transition: 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: #60a5fa !important;
    box-shadow: 0 0 15px rgba(59,130,246,0.6);
}

/* Drag Text */
.stFileUploader label span {
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

/* Limit text */
.stFileUploader small {
    color: #38bdf8 !important;
    font-weight: 600 !important;
}

/* Browse Button */
.stFileUploader button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    border: none !important;
}

/* Uploaded file container */
[data-testid="stFileUploaderFile"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 14px !important;
    padding: 14px !important;
    animation: fadeGlow 0.6s ease-in-out;
}

/* File name */
[data-testid="stFileUploaderFile"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* File size */
[data-testid="stFileUploaderFile"] small {
    color: #60a5fa !important;
    font-weight: 600 !important;
}

/* Upload glow animation */
@keyframes fadeGlow {
    0% { box-shadow: 0 0 0 rgba(59,130,246,0); }
    50% { box-shadow: 0 0 20px rgba(59,130,246,0.6); }
    100% { box-shadow: 0 0 0 rgba(59,130,246,0); }
}

/* Download Button */
.stDownloadButton>button {
    background: linear-gradient(90deg, #10b981, #059669);
    color: white;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
    border: none;
    font-size: 15px;
}

/* Metric Cards */
.metric-card {
    background-color: #111827;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #334155;
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #3b82f6;
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
    "<div class='subtitle'>Advanced Machine Learning Models for Credit Default Prediction</div>",
    unsafe_allow_html=True
)

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
        "Upload Test CSV File",
        type=["csv"]
    )

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

    st.success("✅ File uploaded successfully and ready for evaluation.")

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

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("📊 Performance Metrics")

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

        st.markdown("<hr>", unsafe_allow_html=True)
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

        ax.set_facecolor("#050a18")
        fig.patch.set_facecolor("#050a18")

        st.pyplot(fig)

st.markdown("<div class='footer'>Enterprise Credit Risk Analytics Platform</div>", unsafe_allow_html=True)
