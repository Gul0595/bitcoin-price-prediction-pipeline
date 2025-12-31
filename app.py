import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

def forecast_next_days(df, model, scaler, days):
    future_predictions = []
    temp_df = df.copy()

    for _ in range(days):
        # last row se features lo
        X_last = temp_df.drop(columns=["date", "close"]).iloc[-1:]
        X_last_scaled = scaler.transform(X_last)

        # prediction
        next_price = model.predict(X_last_scaled)[0]
        future_predictions.append(next_price)

        # next day ka row banao
        next_row = temp_df.iloc[-1].copy()
        next_row["close"] = next_price
        next_row["date"] = next_row["date"] + pd.Timedelta(days=1)

        temp_df = pd.concat(
            [temp_df, next_row.to_frame().T],
            ignore_index=True
        )

    return future_predictions

def train_model_with_mlflow(df):
    X = df.drop(columns=["date", "close"])
    y = df["close"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlflow.set_experiment("Bitcoin Price Prediction")

    with mlflow.start_run():
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        rmse = (mean_squared_error(y_test, preds)) ** 0.5

        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, "model")

    return model, scaler, rmse

# Load model & scaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

@st.cache_resource
def train_model():
    df = pd.read_csv("data/processed/btc_feature_data.csv")
    df["date"] = pd.to_datetime(df["date"])

    X = df.drop(columns=["date", "close"])
    y = df["close"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model()


# Load feature data
df = pd.read_csv("data/processed/btc_feature_data.csv")
df["date"] = pd.to_datetime(df["date"])

st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

st.title("📈 Bitcoin Price Prediction Dashboard")

# Sidebar
st.sidebar.header("User Controls")
forecast_days = st.sidebar.slider(
    "Select number of days to forecast",
    min_value=7,
    max_value=60,
    value=30
)

future_prices = forecast_next_days(
    df,
    model,
    scaler,
    forecast_days
)

future_dates = pd.date_range(
    start=df["date"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

st.subheader("🔮 Bitcoin Price Forecast")

fig, ax = plt.subplots(figsize=(12, 5))

# last 100 days ka historical
ax.plot(
    df["date"].tail(100),
    df["close"].tail(100),
    label="Historical Price"
)

# future prediction
ax.plot(
    future_dates,
    future_prices,
    label="Predicted Price",
    linestyle="--"
)

ax.set_xlabel("Date")
ax.set_ylabel("BTC Price (USD)")
ax.legend()

st.pyplot(fig)

st.subheader("🧪 Train Model with MLflow")

if st.button("Train Model & Track Experiment"):
    with st.spinner("Training model and logging to MLflow..."):
        model, scaler, rmse = train_model_with_mlflow(df)

    st.success("Training completed successfully!")
    st.metric("RMSE", f"{rmse:.2f}")


# Select recent data
recent_df = df.tail(forecast_days)

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

