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
# CLEAN ENTERPRISE DARK THEME (FIXED SPACING)
# ---------------------------------------------------
st.markdown("""
<style>

/* Reduce default top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Background */
.stApp {
    background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
    color: #e5e7eb;
}

/* Headers */
h1 {
    font-size: 38px !important;
    font-weight: 700;
    margin-bottom: 0.5rem !important;
}

h2 {
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    font-weight: 600;
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
    margin: 1.5rem 0;
}

/* SELECTBOX FIX */
.stSelectbox > div > div {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    min-height: 50px !important;
    display: flex !important;
    align-items: center !important;
    padding-left: 12px !important;
}

/* Make dropdown arrow visible */
.stSelectbox svg {
    fill: #60a5fa !important;
}

/* FILE UPLOADER FIX */
.stFileUploader > div {
    background-color: #1e293b !important;
    border: 2px dashed #3b82f6 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Browse Button */
.stFileUploader button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
}

/* Download button */
.stDownloadButton>button {
    background: linear-gradient(90deg, #10b981, #059669);
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    border: none;
}

/* METRIC CARDS */
.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #334155;
}

.metric-title {
    font-size: 14px;
    color: #94a3b8;
}

.metric-value {
    font-size: 26px;
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
# --------------------------------------
