import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
st.write("App Started Successfully")
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Data Risk Intelligence System", layout="wide")

st.title("🛡 Unified Data Risk Intelligence System")
st.write("Detect Data Quality Issues, Leakage Risks, ML & DL Anomalies")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =============================
    # DATA QUALITY
    # =============================
    st.subheader("🔍 Data Quality Analysis")

    missing_percent = df.isnull().mean() * 100
    missing_score = missing_percent.mean()
    
    duplicate_count = df.duplicated().sum()

    def detect_outliers_iqr(data):
        outlier_count = 0
        for col in data.select_dtypes(include=np.number).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count += ((data[col] < lower) | (data[col] > upper)).sum()
        return outlier_count

    outlier_count = detect_outliers_iqr(df)

    quality_score = (
        0.4 * missing_score +
        0.3 * duplicate_count +
        0.3 * outlier_count
    )

    quality_norm = quality_score / (df.shape[0] + 1)

    st.write("Quality Score:", round(quality_norm * 100, 2))

    # =============================
    # DATA LEAKAGE
    # =============================
    st.subheader("🔐 Data Leakage Detection")

    def detect_sensitive_data(data):
        sensitive_count = 0
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        for col in data.columns:
            for value in data[col].astype(str):
                if re.search(email_pattern, value):
                    sensitive_count += 1
        return sensitive_count

    leakage_count = detect_sensitive_data(df)
    leakage_norm = leakage_count / (df.shape[0] + 1)

    st.write("Leakage Score:", round(leakage_norm * 100, 2))

    # =============================
    # MACHINE LEARNING
    # =============================
    st.subheader("🤖 Machine Learning (Isolation Forest)")

    numeric_df = df.select_dtypes(include=np.number).fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    model_if = IsolationForest(contamination=0.05, random_state=42)
    model_if.fit(scaled_data)

    df["ML_Anomaly"] = model_if.predict(scaled_data)
    ml_anomaly_count = (df["ML_Anomaly"] == -1).sum()
    ml_norm = ml_anomaly_count / (df.shape[0] + 1)

    st.write("ML Anomaly Score:", round(ml_norm * 100, 2))

    # =============================
    # DEEP LEARNING
    # =============================
    st.subheader("🔥 Deep Learning (PyTorch Autoencoder)")

    scaler_dl = MinMaxScaler()
    scaled_dl_data = scaler_dl.fit_transform(numeric_df)
    data_tensor = torch.tensor(scaled_dl_data, dtype=torch.float32)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim),
                nn.Sigmoid()
            )
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    input_dim = scaled_dl_data.shape[1]
    model = Autoencoder(input_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, data_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        reconstructed = model(data_tensor)
        mse = torch.mean((data_tensor - reconstructed) ** 2, dim=1)

    threshold = torch.quantile(mse, 0.95)
    deep_anomaly_count = torch.sum(mse > threshold).item()
    dl_norm = deep_anomaly_count / (df.shape[0] + 1)

    st.write("Deep Learning Score:", round(dl_norm * 100, 2))

    # =============================
    # FINAL RISK FUSION
    # =============================
    st.subheader("🚨 Final Risk Assessment")

    final_risk_score = (
        0.3 * quality_norm +
        0.2 * leakage_norm +
        0.2 * ml_norm +
        0.3 * dl_norm
    ) * 100

    st.metric("Final Risk Score", round(final_risk_score, 2))



    if final_risk_score < 30:
        st.success("LOW RISK")
    elif final_risk_score < 70:
        st.warning("MEDIUM RISK")
    else:
        st.error("HIGH RISK 🚨")