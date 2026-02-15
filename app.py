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
# ENHANCED DARK THEME (Professional Fintech Look)
# ---------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    color: #e5e7eb;
}

/* Headers */
h1 {
    font-size: 40px !important;
    font-weight: 700;
}
h2 {
    margin-top: 40px !important;
}

/* Subtitle */
.subtitle {
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Divider */
hr {
    border: 1px solid #1f2937;
}

/* Primary Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
    transform: translateY(-2px);
}

/* Download button */
.stDownloadButton>button {
    background: linear-gradient(90deg, #10b981, #059669);
    color: white;
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: 600;
    border: none;
}

.stDownloadButton>button:hover {
    background: linear-gradient(90deg, #059669, #047857);
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #1f2937;
    color: white;
    border-radius: 8px;
}

/* File uploader */
.stFileUploader > div {
    background-color: #1f2937;
    color: white;
    border-radius: 8px;
    border: 1px solid #374151;
}

/* Metric Cards */
.metric-card {
    background-color: #1f2937;
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #374151;
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #2563eb;
}

.metric-title {
    font-size: 14px;
    color: #9ca3af;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #60a5fa;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 40px;
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

st.markdown("---")

# ---------------------------------------------------
# SAMPLE DATASET FIRST
# ---------------------------------------------------
st.header("📥 Sample Dataset")

with open("data/UCI_Credit_Card.csv", "rb") as file:
    st.download_button(
        label="Download Credit Card Dataset",
        data=file,
        file_name="UCI_Credit_Card.csv",
        mime="text/csv"
    )

st.markdown("---")

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

        st.markdown("---")
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

        st.markdown("---")
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

        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")

        st.pyplot(fig)

        st.markdown("---")
        st.header("📋 Model Summary")

        summary_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "MCC"],
            "Value": [acc, prec, rec, f1, auc, mcc]
        })

        st.dataframe(summary_df, use_container_width=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("<div class='footer'>Enterprise Credit Risk Analytics Platform | Streamlit Deployment</div>", unsafe_allow_html=True)
