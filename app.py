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
# PREMIUM DARK ENTERPRISE THEME
# ---------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
    color: #f1f5f9;
}

/* Reduce extra padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headers */
h1 {
    font-size: 38px !important;
    font-weight: 700;
    margin-bottom: 0.5rem !important;
    color: #ffffff !important;
}

h2 {
    font-weight: 600 !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    color: #e2e8f0 !important;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 1.5rem;
}

/* Divider */
hr {
    border: 1px solid #1f2937;
    margin: 1.8rem 0;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    min-height: 52px !important;
    padding-left: 14px !important;
}

.stSelectbox svg {
    fill: #60a5fa !important;
}

/* File uploader container */
.stFileUploader > div {
    background: linear-gradient(135deg, #1e293b, #1a2438);
    border: 2px dashed #3b82f6 !important;
    border-radius: 14px !important;
    padding: 28px !important;
    transition: 0.3s ease;
}

/* File uploader label */
.stFileUploader label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #60a5fa !important;
}

/* Drag text */
.stFileUploader div div span {
    color: #cbd5e1 !important;
    font-size: 14px !important;
}

/* Browse button */
.stFileUploader button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
}

/* Uploaded file name */
.stFileUploader small {
    color: #94a3b8 !important;
}

/* Download button */
.stDownloadButton>button {
    background: linear-gradient(90deg, #10b981, #059669);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
}

/* Metric cards */
.metric-card {
    background-color: #1e293b;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #334155;
}

.metric-title {
    font-size: 14px;
    color: #94a3b8;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
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
    "<div class='subtitle'>Advanced machine learning models for credit default prediction</div>",
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
    st.markdown("### 📂 Upload Test CSV File")
    uploaded_file = st.file_uploader(
        "",
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

        ax.set_facecolor("#0b1220")
        fig.patch.set_facecolor("#0b1220")

        st.pyplot(fig)

st.markdown("<div class='footer'>Enterprise Credit Risk Analytics Platform</div>", unsafe_allow_html=True)
