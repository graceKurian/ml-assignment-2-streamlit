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
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Enterprise Credit Risk Dashboard", layout="wide")

# ---------------------------------------------------
# ENTERPRISE DARK THEME + ANIMATIONS
# ---------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: #e2e8f0;
}

.main-container {
    max-width: 1200px;
    margin: auto;
    animation: fadeIn 0.8s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

.header-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.header-subtitle {
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.section-box {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 25px;
    border: 1px solid #334155;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #334155;
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #3b82f6;
}

.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #3b82f6;
}

.metric-label {
    font-size: 14px;
    color: #94a3b8;
}

.footer {
    text-align: center;
    font-size: 13px;
    color: #64748b;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown("<div class='header-title'>🏦 Enterprise Credit Risk Analytics</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Advanced Machine Learning Models for Credit Default Prediction</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Model Selection
# ---------------------------------------------------
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("### 🔎 Select Model")

model_name = st.selectbox(
    "",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ]
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Dataset Download
# ---------------------------------------------------
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("### 📥 Download Sample Dataset")

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
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("### 📂 Upload Test CSV File")
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
# Evaluation Logic
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

        # ---------------------------------------------------
        # Metrics Section
        # ---------------------------------------------------
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("### 📊 Performance Metrics")

        col1, col2, col3 = st.columns(3)

        metrics = [
            ("Accuracy", acc),
            ("Precision", prec),
            ("Recall", rec),
            ("F1 Score", f1),
            ("AUC", auc),
            ("MCC", mcc),
        ]

        for col, (label, value) in zip([col1, col1, col2, col2, col3, col3], metrics):
            col.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-value'>{value:.4f}</div>
                    <div class='metric-label'>{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("### 📈 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="coolwarm",
            cbar=False,
            linewidths=1,
            linecolor="#1e293b",
            square=True,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------
        # Model Comparison Table
        # ---------------------------------------------------
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Model Performance Summary")

        comparison_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "MCC"],
            "Value": [acc, prec, rec, f1, auc, mcc]
        })

        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("<div class='footer'>Enterprise Credit Risk Analytics Platform | Powered by Streamlit & Scikit-learn</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
