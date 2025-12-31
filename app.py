import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model & scaler
model = joblib.load("models/xgboost_btc_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load feature data
df = pd.read_csv("data/processed/btc_feature_data.csv")
df["date"] = pd.to_datetime(df["date"])

st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

st.title("📈 Bitcoin Price Prediction Dashboard")

# Sidebar
st.sidebar.header("Prediction Settings")
days = st.sidebar.slider("Number of days to visualize", 30, 180, 90)

# Select recent data
recent_df = df.tail(days)

X = recent_df.drop(columns=["date", "close"])
X_scaled = scaler.transform(X)

# Predict
recent_df["predicted_price"] = model.predict(X_scaled)

# Plot
st.subheader("Actual vs Predicted Bitcoin Price")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(recent_df["date"], recent_df["close"], label="Actual Price")
ax.plot(recent_df["date"], recent_df["predicted_price"], label="Predicted Price")
ax.set_xlabel("Date")
ax.set_ylabel("BTC Price (USD)")
ax.legend()

st.pyplot(fig)

# Metrics
st.subheader("Latest Prediction")
st.metric("Actual Price", f"${recent_df['close'].iloc[-1]:,.2f}")
st.metric("Predicted Price", f"${recent_df['predicted_price'].iloc[-1]:,.2f}")

st.caption("⚠️ This model is for educational purposes only.")

