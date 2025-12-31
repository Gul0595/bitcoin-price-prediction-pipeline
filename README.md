# 📊 Bitcoin Price Prediction using Machine Learning

## 📌 Project Overview
This project builds an end-to-end data science pipeline to analyze and predict Bitcoin prices using historical market data. The pipeline covers data ingestion, preprocessing, feature engineering, model benchmarking, and deployment using Streamlit.

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- CoinGecko API

## 🔄 Pipeline Architecture
1. Data Ingestion (CoinGecko API)
2. Data Preprocessing
3. Feature Engineering (MA, RSI, Lag Features, Volatility)
4. Model Training & Evaluation
5. Model Deployment (Streamlit)

## 📈 Features Engineered
- Moving Averages (MA7, MA30)
- Relative Strength Index (RSI)
- Lagged Prices (1, 3, 7 days)
- Rolling Volatility
- Daily Returns

## 🤖 Models Used
- Linear Regression (Baseline)
- Random Forest Regressor
- XGBoost Regressor (Best Performer)

## 📊 Evaluation Metrics
- MAE
- RMSE

XGBoost achieved the lowest error due to its ability to model non-linear price movements.

## 🚀 Deployment
A Streamlit web application visualizes actual vs predicted Bitcoin prices.

## ⚠️ Disclaimer
This project is for educational purposes only and should not be used for financial decisions.

## 📁 Project Structure
